import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sonnet as snt
import itertools
from orderedattrdict import AttrDict

from dps import cfg
from dps.utils import Param
from dps.utils.tf import (
    build_scheduled_value, FIXED_COLLECTION, ScopedFunction,
    tf_shape, apply_object_wise, tf_binomial_coefficient
)

from auto_yolo.tf_ops import render_sprites, resampler_edge
from auto_yolo.models.core import (
    concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl, tf_safe_log,
    coords_to_image_space)

Normal = tfp.distributions.Normal


class ObjectLayer(ScopedFunction):
    object_shape = Param()
    A = Param()
    training_wheels = Param()
    noisy = Param()
    eval_noisy = Param()
    edge_resampler = Param()
    obj_concrete_temp = Param(help="Higher values -> smoother")
    obj_temp = Param(help="Higher values -> more uniform")

    def __init__(self, scope=None, **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.training_wheels = build_scheduled_value(self.training_wheels, "training_wheels")
        self.obj_concrete_temp = build_scheduled_value(self.obj_concrete_temp, "obj_concrete_temp")
        self.obj_temp = build_scheduled_value(self.obj_temp, "obj_temp")

    def std_nonlinearity(self, std_logit):
        # return tf.exp(std)
        return (
            self._noisy * 2 * tf.nn.sigmoid(tf.clip_by_value(std_logit, -10, 10))
            + (1 - self._noisy) * tf.zeros_like(std_logit)
        )

    def z_nonlinearity(self, z_logit):
        return tf.nn.sigmoid(tf.clip_by_value(z_logit, -10, 10))

    @property
    def _noisy(self):
        return (
            self.float_is_training * tf.to_float(self.noisy)
            + (1 - self.float_is_training) * tf.to_float(self.eval_noisy)
        )

    def _compute_obj_kl(self, tensors, existing_objects=None, simple=False):
        # --- compute obj_kl ---
        if simple:
            prior = self._independent_prior()
            return concrete_binary_sample_kl(
                tensors["obj_pre_sigmoid"],
                tensors["obj_log_odds"], self.obj_concrete_temp,
                prior["obj_log_odds"], self.obj_concrete_temp)

        obj_pre_sigmoid = tensors["obj_pre_sigmoid"]
        obj_log_odds = tensors["obj_log_odds"]
        obj_prob = tensors["obj_prob"]
        obj = tensors["obj"]
        batch_size, n_objects, _ = tf_shape(obj)

        max_n_objects = n_objects

        if existing_objects is not None:
            _, n_existing_objects, _ = tf_shape(existing_objects)
            existing_objects = tf.reshape(existing_objects, (batch_size, n_existing_objects))
            max_n_objects += n_existing_objects

        count_support = tf.range(max_n_objects+1, dtype=tf.float32)

        if self.count_prior_dist is not None:
            if self.count_prior_dist is not None:
                assert len(self.count_prior_dist) == (max_n_objects + 1)
            count_distribution = tf.constant(self.count_prior_dist, dtype=tf.float32)
        else:
            count_prior_prob = tf.nn.sigmoid(self.count_prior_log_odds)
            count_distribution = count_prior_prob ** count_support

        normalizer = tf.reduce_sum(count_distribution)
        count_distribution = count_distribution / tf.maximum(normalizer, 1e-6)
        count_distribution = tf.tile(count_distribution[None, :], (batch_size, 1))

        if existing_objects is not None:
            count_so_far = tf.reduce_sum(tf.round(existing_objects), axis=1, keepdims=True)

            count_distribution = (
                count_distribution
                * tf_binomial_coefficient(count_support, count_so_far)
                * tf_binomial_coefficient(max_n_objects - count_support, n_existing_objects - count_so_far)
            )

            normalizer = tf.reduce_sum(count_distribution, axis=1, keepdims=True)
            count_distribution = count_distribution / tf.maximum(normalizer, 1e-6)
        else:
            count_so_far = tf.zeros((batch_size, 1), dtype=tf.float32)

        obj_kl = []
        for i in range(n_objects):
            p_z_given_Cz_raw = (count_support[None, :] - count_so_far) / (max_n_objects - i)
            p_z_given_Cz = tf.clip_by_value(p_z_given_Cz_raw, 0.0, 1.0)

            # Doing this instead of 1 - p_z_given_Cz seems to be more numerically stable.
            inv_p_z_given_Cz_raw = (max_n_objects - i - count_support[None, :] + count_so_far) / (max_n_objects - i)
            inv_p_z_given_Cz = tf.clip_by_value(inv_p_z_given_Cz_raw, 0.0, 1.0)

            p_z = tf.reduce_sum(count_distribution * p_z_given_Cz, axis=1, keepdims=True)

            if self.use_concrete_kl:
                prior_log_odds = tf_safe_log(p_z) - tf_safe_log(1-p_z)
                _obj_kl = concrete_binary_sample_kl(
                    obj_pre_sigmoid[:, i, :],
                    obj_log_odds[:, i, :], self.obj_concrete_temp,
                    prior_log_odds, self.obj_concrete_temp,
                )
            else:
                prob = obj_prob[:, i, :]

                _obj_kl = (
                    prob * (tf_safe_log(prob) - tf_safe_log(p_z))
                    + (1-prob) * (tf_safe_log(1-prob) - tf_safe_log(1-p_z))
                )

            obj_kl.append(_obj_kl)

            sample = tf.to_float(obj[:, i, :] > 0.5)
            mult = sample * p_z_given_Cz + (1-sample) * inv_p_z_given_Cz
            raw_count_distribution = mult * count_distribution
            normalizer = tf.reduce_sum(raw_count_distribution, axis=1, keepdims=True)
            normalizer = tf.maximum(normalizer, 1e-6)

            # invalid = tf.logical_and(p_z_given_Cz_raw > 1, count_distribution > 1e-8)
            # float_invalid = tf.cast(invalid, tf.float32)
            # diagnostic = tf.stack(
            #     [float_invalid, p_z_given_Cz, count_distribution, mult, raw_count_distribution], axis=-1)

            # assert_op = tf.Assert(
            #     tf.reduce_all(tf.logical_not(invalid)),
            #     [invalid, diagnostic, count_so_far, sample, tf.constant(i, dtype=tf.float32)],
            #     summarize=100000)

            count_distribution = raw_count_distribution / normalizer
            count_so_far += sample

            # this avoids buildup of inaccuracies that can cause problems in computing p_z_given_Cz_raw
            count_so_far = tf.round(count_so_far)

        obj_kl = tf.reshape(tf.concat(obj_kl, axis=1), (batch_size, n_objects, 1))

        return obj_kl


class ObjectRenderer(ScopedFunction):
    object_shape = Param()

    color_logit_scale = Param()
    alpha_logit_scale = Param()
    alpha_logit_bias = Param()
    anchor_box = Param()
    importance_temp = Param()

    def __init__(self, scope=None, **kwargs):
        self.anchor_box = np.array(self.anchor_box)
        super().__init__(scope=scope, **kwargs)

    def _call(self, objects, background, is_training, appearance_only=False):
        if not self.initialized:
            self.image_depth = tf_shape(background)[-1]

        self.maybe_build_subnet("object_decoder")

        # --- compute sprite appearance from attr using object decoder ---

        appearance_logit = apply_object_wise(
            self.object_decoder, objects.attr,
            output_size=self.object_shape + (self.image_depth+1,),
            is_training=is_training)

        appearance_logit = appearance_logit * ([self.color_logit_scale] * self.image_depth + [self.alpha_logit_scale])
        appearance_logit = appearance_logit + ([0.] * self.image_depth + [self.alpha_logit_bias])

        appearance = tf.nn.sigmoid(tf.clip_by_value(appearance_logit, -10., 10.))

        if appearance_only:
            return dict(appearance=appearance)

        appearance_for_output = appearance

        batch_size, *obj_leading_shape, _, _, _ = tf_shape(appearance)
        n_objects = np.prod(obj_leading_shape)
        appearance = tf.reshape(
            appearance, (batch_size, n_objects, *self.object_shape, self.image_depth+1))

        obj_colors, obj_alpha = tf.split(appearance, [self.image_depth, 1], axis=-1)

        obj_alpha *= tf.reshape(objects.obj, (batch_size, n_objects, 1, 1, 1))

        z = tf.reshape(objects.z, (batch_size, n_objects, 1, 1, 1))
        obj_importance = tf.maximum(obj_alpha * z / self.importance_temp, 0.01)

        object_maps = tf.concat([obj_colors, obj_alpha, obj_importance], axis=-1)

        *_, image_height, image_width, _ = tf_shape(background)

        yt, xt, ys, xs = coords_to_image_space(
            objects.yt, objects.xt, objects.ys, objects.xs,
            (image_height, image_width), self.anchor_box, top_left=True)

        scales = tf.concat([ys, xs], axis=-1)
        scales = tf.reshape(scales, (batch_size, n_objects, 2))

        offsets = tf.concat([yt, xt], axis=-1)
        offsets = tf.reshape(offsets, (batch_size, n_objects, 2))

        # --- Compose images ---

        n_objects_per_image = tf.fill((batch_size,), int(n_objects))

        output = render_sprites.render_sprites(
            object_maps,
            n_objects_per_image,
            scales,
            offsets,
            background
        )

        return dict(
            appearance=appearance_for_output,
            output=output)


class GridObjectLayer(ObjectLayer):
    n_passthrough_features = Param()
    n_lookback = Param()

    yx_prior_mean = Param()
    yx_prior_std = Param()
    min_yx = Param()
    max_yx = Param()

    hw_prior_mean = Param()
    hw_prior_std = Param()
    min_hw = Param()
    max_hw = Param()
    anchor_box = Param()

    z_prior_mean = Param()
    z_prior_std = Param()

    attr_prior_mean = Param()
    attr_prior_std = Param()

    use_concrete_kl = Param()
    count_prior_log_odds = Param()
    count_prior_dist = Param()
    n_objects_per_cell = Param()

    edge_weights = None

    def __init__(self, pixels_per_cell, scope=None, **kwargs):
        super(GridObjectLayer, self).__init__(scope=scope, **kwargs)

        self.pixels_per_cell = pixels_per_cell

        self.B = self.n_objects_per_cell

        if isinstance(self.count_prior_dist, str):
            self.count_prior_dist = eval(self.count_prior_dist)

        self.count_prior_log_odds = build_scheduled_value(self.count_prior_log_odds, "count_prior_log_odds")

        self.yx_prior_mean = build_scheduled_value(self.yx_prior_mean, "yx_prior_mean")
        self.yx_prior_std = build_scheduled_value(self.yx_prior_std, "yx_prior_std")

        self.hw_prior_mean = build_scheduled_value(self.hw_prior_mean, "hw_prior_mean")
        self.hw_prior_std = build_scheduled_value(self.hw_prior_std, "hw_prior_std")

        self.anchor_box = np.array(self.anchor_box)

    def _independent_prior(self):
        return dict(
            cell_y_logit_mean=self.yx_prior_mean,
            cell_x_logit_mean=self.yx_prior_mean,
            height_logit_mean=self.hw_prior_mean,
            width_logit_mean=self.hw_prior_mean,
            attr_mean=self.attr_prior_mean,
            z_logit_mean=self.z_prior_mean,

            cell_y_logit_std=self.yx_prior_std,
            cell_x_logit_std=self.yx_prior_std,
            height_logit_std=self.hw_prior_std,
            width_logit_std=self.hw_prior_std,
            attr_std=self.attr_prior_std,
            z_logit_std=self.z_prior_std,

            obj_log_odds=self.count_prior_log_odds,
        )

    def compute_kl(self, tensors, prior=None, do_obj=True):
        if prior is None:
            prior = self._independent_prior()

        def normal_kl(name):
            loc_name = name + "_mean"
            scale_name = name + "_std"
            _post = Normal(loc=tensors[loc_name], scale=tensors[scale_name])
            _prior = Normal(loc=prior[loc_name], scale=prior[scale_name])
            return _post.kl_divergence(_prior)

        kl = dict(
            cell_y_kl=normal_kl("cell_y_logit"),
            cell_x_kl=normal_kl("cell_x_logit"),
            height_kl=normal_kl("height_logit"),
            width_kl=normal_kl("width_logit"),
            z_kl=normal_kl("z_logit"),
            attr_kl=normal_kl("attr")
        )

        if do_obj:
            kl['obj_kl'] = self._compute_obj_kl(tensors)

        return kl

    def _build_box(self, box_params, is_training, hw=None):
        mean, log_std = tf.split(box_params, 2, axis=-1)

        std = self.std_nonlinearity(log_std)

        mean = self.training_wheels * tf.stop_gradient(mean) + (1-self.training_wheels) * mean
        std = self.training_wheels * tf.stop_gradient(std) + (1-self.training_wheels) * std

        cell_y_mean, cell_x_mean, height_mean, width_mean = tf.split(mean, 4, axis=-1)
        cell_y_std, cell_x_std, height_std, width_std = tf.split(std, 4, axis=-1)

        cell_y_logit = Normal(loc=cell_y_mean, scale=cell_y_std).sample()
        cell_x_logit = Normal(loc=cell_x_mean, scale=cell_x_std).sample()
        height_logit = Normal(loc=height_mean, scale=height_std).sample()
        width_logit = Normal(loc=width_mean, scale=width_std).sample()

        # --- cell y/x transform ---

        cell_y = tf.nn.sigmoid(tf.clip_by_value(cell_y_logit, -10, 10))
        cell_x = tf.nn.sigmoid(tf.clip_by_value(cell_x_logit, -10, 10))

        assert self.max_yx > self.min_yx

        cell_y = float(self.max_yx - self.min_yx) * cell_y + self.min_yx
        cell_x = float(self.max_yx - self.min_yx) * cell_x + self.min_yx

        # --- height/width transform ---

        height = tf.nn.sigmoid(tf.clip_by_value(height_logit, -10, 10))
        width = tf.nn.sigmoid(tf.clip_by_value(width_logit, -10, 10))
        assert self.max_hw > self.min_hw

        height = float(self.max_hw - self.min_hw) * height + self.min_hw
        width = float(self.max_hw - self.min_hw) * width + self.min_hw

        local_box = tf.concat([cell_y, cell_x, height, width], axis=-1)

        # --- Compute image-normalized box parameters ---

        ys = height
        xs = width

        # box center normalized to anchor box
        if hw is None:
            w, h = tf.meshgrid(
                tf.range(self.W, dtype=tf.float32),
                tf.range(self.H, dtype=tf.float32))
            h = h[None, :, :, None]
            w = w[None, :, :, None]
            yt = (self.pixels_per_cell[0] / self.anchor_box[0]) * (cell_y + h)
            xt = (self.pixels_per_cell[1] / self.anchor_box[1]) * (cell_x + w)
        else:
            h, w = hw
            yt = (self.pixels_per_cell[0] / self.anchor_box[0]) * (cell_y + h)
            xt = (self.pixels_per_cell[1] / self.anchor_box[1]) * (cell_x + w)

        normalized_box = tf.concat([yt, xt, ys, xs], axis=-1)

        ys_logit = height_logit
        xs_logit = width_logit

        return dict(
            # "raw" box values
            cell_y=cell_y,
            cell_x=cell_x,
            height=height,
            width=width,
            local_box=local_box,

            cell_y_logit_mean=cell_y_mean,
            cell_x_logit_mean=cell_x_mean,
            height_logit_mean=height_mean,
            width_logit_mean=width_mean,

            cell_y_logit_std=cell_y_std,
            cell_x_logit_std=cell_x_std,
            height_logit_std=height_std,
            width_logit_std=width_std,

            # box center and height/width, in a coordinate frame where (0, 0) is image top-left
            # and (1, 1) is image bottom-right

            # box center and scale with respect to anchor_box
            yt=yt,
            xt=xt,
            ys=ys,
            xs=xs,
            normalized_box=normalized_box,

            ys_logit=ys_logit,
            xs_logit=xs_logit,
        )

    def _build_obj(self, obj_logit, is_training, **kwargs):
        obj_logit = self.training_wheels * tf.stop_gradient(obj_logit) + (1-self.training_wheels) * obj_logit
        obj_log_odds = tf.clip_by_value(obj_logit / self.obj_temp, -10., 10.)

        obj_pre_sigmoid = (
            self._noisy * concrete_binary_pre_sigmoid_sample(obj_log_odds, self.obj_concrete_temp)
            + (1 - self._noisy) * obj_log_odds
        )

        obj = tf.nn.sigmoid(obj_pre_sigmoid)

        return dict(
            obj_log_odds=obj_log_odds,
            obj_prob=tf.nn.sigmoid(obj_log_odds),
            obj_pre_sigmoid=obj_pre_sigmoid,
            obj=obj,
        )

    def _get_sequential_context(self, program, h, w, b, edge_element):
        context = []
        grid_size = 2 * self.n_lookback + 1
        n_grid_locs = int((grid_size**2) / 2)

        # Surrounding locations
        for idx in range(n_grid_locs):
            _i = int(idx / grid_size) + h - self.n_lookback
            _j = int(idx % grid_size) + w - self.n_lookback

            for k in range(self.B):
                if _i < 0 or _j < 0 or _i >= program.shape[0] or _j >= program.shape[1]:
                    context.append(edge_element)
                else:
                    context.append(program[_i, _j, k])

        # Current location, but previous anchor boxes
        offset = -(self.B - 1) + b
        for k in range(self.B-1):
            _k = k + offset
            if _k < 0:
                context.append(edge_element)
            else:
                context.append(program[h, w, _k])

        if context:
            return tf.concat(context, axis=1)
        else:
            return tf.zeros_like(edge_element[:, 0:0])

    def _call(self, inp, inp_features, is_training, is_posterior=True, prop_state=None):
        print("\n" + "-" * 10 + " GridObjectLayer(is_posterior={}) ".format(is_posterior) + "-" * 10)

        # --- set up sub networks and attributes ---

        self.maybe_build_subnet("box_network", builder=cfg.build_lateral, key="box")
        self.maybe_build_subnet("attr_network", builder=cfg.build_lateral, key="attr")
        self.maybe_build_subnet("z_network", builder=cfg.build_lateral, key="z")
        self.maybe_build_subnet("obj_network", builder=cfg.build_lateral, key="obj")

        self.maybe_build_subnet("object_encoder")

        _, H, W, _, _ = tf_shape(inp_features)
        H = int(H)
        W = int(W)

        if not self.initialized:
            # Note this limits the re-usability of this module to images
            # with a fixed shape (the shape of the first image it is used on)
            self.batch_size, self.image_height, self.image_width, self.image_depth = tf_shape(inp)
            self.H = H
            self.W = W
            self.HWB = H*W*self.B
            self.is_training = is_training
            self.float_is_training = tf.to_float(is_training)

        # --- set up the edge element ---

        sizes = [4, self.A, 1, 1]
        sigmoids = [True, False, False, True]
        total_sample_size = sum(sizes)

        if self.edge_weights is None:
            self.edge_weights = tf.get_variable("edge_weights", shape=total_sample_size, dtype=tf.float32)
            if "edge" in self.fixed_weights:
                tf.add_to_collection(FIXED_COLLECTION, self.edge_weights)

        _edge_weights = tf.split(self.edge_weights, sizes, axis=0)
        _edge_weights = [
            (tf.nn.sigmoid(ew) if sigmoid else ew)
            for ew, sigmoid in zip(_edge_weights, sigmoids)]
        edge_element = tf.concat(_edge_weights, axis=0)
        edge_element = tf.tile(edge_element[None, :], (self.batch_size, 1))

        # --- containers for storing built program ---

        program = np.empty((H, W, self.B), dtype=np.object)

        # --- build the program ---

        is_posterior_tf = tf.ones((self.batch_size, 2))
        if is_posterior:
            is_posterior_tf = is_posterior_tf * [1, 0]
        else:
            is_posterior_tf = is_posterior_tf * [0, 1]

        results = []
        for h, w, b in itertools.product(range(H), range(W), range(self.B)):
            built = dict()

            partial_program, features = None, None
            context = self._get_sequential_context(program, h, w, b, edge_element)
            base_features = tf.concat([inp_features[:, h, w, b, :], context, is_posterior_tf], axis=1)

            # --- box ---

            layer_inp = base_features
            n_features = self.n_passthrough_features
            output_size = 8

            network_output = self.box_network(layer_inp, output_size + n_features, self. is_training)
            rep_input, features = tf.split(network_output, (output_size, n_features), axis=1)

            _built = self._build_box(rep_input, self.is_training, hw=(h, w))
            built.update(_built)
            partial_program = built['local_box']

            # --- attr ---

            if is_posterior:
                # --- Get object attributes using object encoder ---

                yt, xt, ys, xs = tf.split(built['normalized_box'], 4, axis=-1)

                yt, xt, ys, xs = coords_to_image_space(
                    yt, xt, ys, xs, (self.image_height, self.image_width), self.anchor_box, top_left=False)

                transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
                warper = snt.AffineGridWarper(
                    (self.image_height, self.image_width), self.object_shape, transform_constraints)

                _boxes = tf.concat([xs, 2*xt - 1, ys, 2*yt - 1], axis=-1)

                grid_coords = warper(_boxes)
                grid_coords = tf.reshape(grid_coords, (self.batch_size, 1, *self.object_shape, 2,))
                if self.edge_resampler:
                    glimpse = resampler_edge.resampler_edge(inp, grid_coords)
                else:
                    glimpse = tf.contrib.resampler.resampler(inp, grid_coords)
                glimpse = tf.reshape(glimpse, (self.batch_size, *self.object_shape, self.image_depth))
            else:
                glimpse = tf.zeros((self.batch_size, *self.object_shape, self.image_depth))

            # Create the object encoder network regardless of is_posterior, otherwise messes with ScopedFunction
            encoded_glimpse = self.object_encoder(glimpse, (1, 1, self.A), self.is_training)
            encoded_glimpse = tf.reshape(encoded_glimpse, (self.batch_size, self.A))

            if not is_posterior:
                encoded_glimpse = tf.zeros_like(encoded_glimpse)

            layer_inp = tf.concat(
                [base_features, features, encoded_glimpse, partial_program], axis=1)
            network_output = self.attr_network(layer_inp, 2 * self.A + n_features, self. is_training)
            attr_mean, attr_log_std, features = tf.split(network_output, (self.A, self.A, n_features), axis=1)

            attr_std = self.std_nonlinearity(attr_log_std)

            attr = Normal(loc=attr_mean, scale=attr_std).sample()

            built.update(attr_mean=attr_mean, attr_std=attr_std, attr=attr, glimpse=glimpse)
            partial_program = tf.concat([partial_program, built['attr']], axis=1)

            # --- z ---

            layer_inp = tf.concat([base_features, features, partial_program], axis=1)
            n_features = self.n_passthrough_features

            network_output = self.z_network(layer_inp, 2 + n_features, self.is_training)
            z_mean, z_log_std, features = tf.split(network_output, (1, 1, n_features), axis=1)
            z_std = self.std_nonlinearity(z_log_std)

            z_mean = self.training_wheels * tf.stop_gradient(z_mean) + (1-self.training_wheels) * z_mean
            z_std = self.training_wheels * tf.stop_gradient(z_std) + (1-self.training_wheels) * z_std
            z_logit = Normal(loc=z_mean, scale=z_std).sample()
            z = self.z_nonlinearity(z_logit)

            built.update(z_logit_mean=z_mean, z_logit_std=z_std, z_logit=z_logit, z=z)
            partial_program = tf.concat([partial_program, built['z']], axis=1)

            # --- obj ---

            layer_inp = tf.concat([base_features, features, partial_program], axis=1)
            rep_input = self.obj_network(layer_inp, 1, self.is_training)

            _built = self._build_obj(rep_input, self.is_training)
            built.update(_built)

            partial_program = tf.concat([partial_program, built['obj']], axis=1)

            # --- final ---

            results.append(built)

            program[h, w, b] = partial_program
            assert program[h, w, b].shape[1] == total_sample_size

        objects = AttrDict()
        for k in results[0]:
            objects[k] = tf.stack([r[k] for r in results], axis=1)

        if prop_state is not None:
            objects.prop_state = tf.tile(prop_state[0:1, None], (self.batch_size, self.HWB, 1))
            objects.prior_prop_state = tf.tile(prop_state[0:1, None], (self.batch_size, self.HWB, 1))

        # --- misc ---

        objects.n_objects = tf.fill((self.batch_size,), self.HWB)
        objects.pred_n_objects = tf.reduce_sum(objects.obj, axis=(1, 2))
        objects.pred_n_objects_hard = tf.reduce_sum(tf.round(objects.obj), axis=(1, 2))

        return objects


class ConvGridObjectLayer(GridObjectLayer):
    def _call(self, inp, inp_features, is_training, is_posterior=True, prop_state=None):
        print("\n" + "-" * 10 + " ConvGridObjectLayer(is_posterior={}) ".format(is_posterior) + "-" * 10)

        # --- set up sub networks and attributes ---

        self.maybe_build_subnet("box_network", builder=cfg.build_conv_lateral, key="box")
        self.maybe_build_subnet("attr_network", builder=cfg.build_conv_lateral, key="attr")
        self.maybe_build_subnet("z_network", builder=cfg.build_conv_lateral, key="z")
        self.maybe_build_subnet("obj_network", builder=cfg.build_conv_lateral, key="obj")

        self.maybe_build_subnet("object_encoder")

        _, H, W, _, n_channels = tf_shape(inp_features)

        if self.B != 1:
            raise Exception("NotImplemented")

        if not self.initialized:
            # Note this limits the re-usability of this module to images
            # with a fixed shape (the shape of the first image it is used on)
            self.batch_size, self.image_height, self.image_width, self.image_depth = tf_shape(inp)
            self.H = H
            self.W = W
            self.HWB = H*W
            self.batch_size = tf.shape(inp)[0]
            self.is_training = is_training
            self.float_is_training = tf.to_float(is_training)

        inp_features = tf.reshape(inp_features, (self.batch_size, H, W, n_channels))
        is_posterior_tf = tf.ones_like(inp_features[..., :2])
        if is_posterior:
            is_posterior_tf = is_posterior_tf * [1, 0]
        else:
            is_posterior_tf = is_posterior_tf * [0, 1]

        objects = AttrDict()

        base_features = tf.concat([inp_features, is_posterior_tf], axis=-1)

        # --- box ---

        layer_inp = base_features
        n_features = self.n_passthrough_features
        output_size = 8

        network_output = self.box_network(layer_inp, output_size + n_features, self.is_training)
        rep_input, features = tf.split(network_output, (output_size, n_features), axis=-1)

        _objects = self._build_box(rep_input, self.is_training)
        objects.update(_objects)

        # --- attr ---

        if is_posterior:
            # --- Get object attributes using object encoder ---

            yt, xt, ys, xs = tf.split(objects['normalized_box'], 4, axis=-1)

            yt, xt, ys, xs = coords_to_image_space(
                yt, xt, ys, xs, (self.image_height, self.image_width), self.anchor_box, top_left=False)

            transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
            warper = snt.AffineGridWarper(
                (self.image_height, self.image_width), self.object_shape, transform_constraints)

            _boxes = tf.concat([xs, 2*xt - 1, ys, 2*yt - 1], axis=-1)
            _boxes = tf.reshape(_boxes, (self.batch_size*H*W, 4))
            grid_coords = warper(_boxes)
            grid_coords = tf.reshape(grid_coords, (self.batch_size, H, W, *self.object_shape, 2,))

            if self.edge_resampler:
                glimpse = resampler_edge.resampler_edge(inp, grid_coords)
            else:
                glimpse = tf.contrib.resampler.resampler(inp, grid_coords)
        else:
            glimpse = tf.zeros((self.batch_size, H, W, *self.object_shape, self.image_depth))

        # Create the object encoder network regardless of is_posterior, otherwise messes with ScopedFunction
        encoded_glimpse = apply_object_wise(
            self.object_encoder, glimpse, n_trailing_dims=3, output_size=self.A, is_training=self.is_training)

        if not is_posterior:
            encoded_glimpse = tf.zeros_like(encoded_glimpse)

        layer_inp = tf.concat([base_features, features, encoded_glimpse, objects['local_box']], axis=-1)
        network_output = self.attr_network(layer_inp, 2 * self.A + n_features, self.is_training)
        attr_mean, attr_log_std, features = tf.split(network_output, (self.A, self.A, n_features), axis=-1)

        attr_std = self.std_nonlinearity(attr_log_std)

        attr = Normal(loc=attr_mean, scale=attr_std).sample()

        objects.update(attr_mean=attr_mean, attr_std=attr_std, attr=attr, glimpse=glimpse)

        # --- z ---

        layer_inp = tf.concat([base_features, features, objects['local_box'], objects['attr']], axis=-1)
        n_features = self.n_passthrough_features

        network_output = self.z_network(layer_inp, 2 + n_features, self.is_training)
        z_mean, z_log_std, features = tf.split(network_output, (1, 1, n_features), axis=-1)
        z_std = self.std_nonlinearity(z_log_std)

        z_mean = self.training_wheels * tf.stop_gradient(z_mean) + (1-self.training_wheels) * z_mean
        z_std = self.training_wheels * tf.stop_gradient(z_std) + (1-self.training_wheels) * z_std
        z_logit = Normal(loc=z_mean, scale=z_std).sample()
        z = self.z_nonlinearity(z_logit)

        objects.update(z_logit_mean=z_mean, z_logit_std=z_std, z_logit=z_logit, z=z)

        # --- obj ---

        layer_inp = tf.concat([base_features, features, objects['local_box'], objects['attr'], objects['z']], axis=-1)
        rep_input = self.obj_network(layer_inp, 1, self.is_training)

        _objects = self._build_obj(rep_input, self.is_training)
        objects.update(_objects)

        # --- final ---

        _objects = AttrDict()
        for k, v in objects.items():
            _, _, _, *trailing_dims = tf_shape(v)
            _objects[k] = tf.reshape(v, (self.batch_size, self.HWB, *trailing_dims))
        objects = _objects

        if prop_state is not None:
            objects.prop_state = tf.tile(prop_state[0:1, None], (self.batch_size, self.HWB, 1))
            objects.prior_prop_state = tf.tile(prop_state[0:1, None], (self.batch_size, self.HWB, 1))

        # --- misc ---

        objects.n_objects = tf.fill((self.batch_size,), self.HWB)
        objects.pred_n_objects = tf.reduce_sum(objects.obj, axis=(1, 2))
        objects.pred_n_objects_hard = tf.reduce_sum(tf.round(objects.obj), axis=(1, 2))

        return objects
