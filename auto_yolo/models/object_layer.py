import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import collections
import sonnet as snt
import itertools

from dps import cfg
from dps.utils import Param
from dps.utils.tf import build_scheduled_value, FIXED_COLLECTION, ScopedFunction

from auto_yolo.tf_ops import render_sprites, resampler_edge
from auto_yolo.models.core import (
    concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl, tf_safe_log)

Normal = tfp.distributions.Normal


class ObjectLayer(ScopedFunction):
    def std_nonlinearity(self, std_logit):
        # return tf.exp(std)
        std = 2 * tf.nn.sigmoid(tf.clip_by_value(std_logit, -10, 10))
        if not self.noisy:
            std = tf.zeros_like(std)
        return std

    def z_nonlinearity(self, z_logits):
        return 4 * tf.nn.sigmoid(tf.clip_by_value(z_logits, -10, 10))

    def z_nonlinearity_inverse(self, z):
        p = tf.clip_by_value(z / 4., 1e-6, 1-1e-6)
        return -tf.log(1. / p - 1.)


class GridObjectLayer(ObjectLayer):
    object_shape = Param()
    n_passthrough_features = Param()
    training_wheels = Param()
    n_lookback = Param()

    yx_prior_mean = Param()
    yx_prior_std = Param()
    min_yx = Param()
    max_yx = Param()

    hw_prior_mean = Param()
    hw_prior_std = Param()
    min_hw = Param()
    max_hw = Param()
    anchor_boxes = Param()

    z_prior_mean = Param()
    z_prior_std = Param()

    attr_prior_mean = Param()
    attr_prior_std = Param()
    A = Param()
    noisy = Param()

    use_concrete_kl = Param()
    count_prior_log_odds = Param()
    count_prior_dist = Param()
    obj_concrete_temp = Param(help="Higher values -> smoother")
    obj_temp = Param(help="Higher values -> more uniform")
    obj_logit_scale = Param()
    alpha_logit_scale = Param()
    alpha_logit_bias = Param()

    edge_weights = None

    def __init__(self, pixels_per_cell, scope=None, **kwargs):
        super(GridObjectLayer, self).__init__(scope=scope, **kwargs)

        self.pixels_per_cell = pixels_per_cell

        self.B = len(self.anchor_boxes)

        if isinstance(self.count_prior_dist, str):
            self.count_prior_dist = eval(self.count_prior_dist)

        self.count_prior_log_odds = build_scheduled_value(
            self.count_prior_log_odds, "count_prior_log_odds")
        self.obj_concrete_temp = build_scheduled_value(self.obj_concrete_temp, "obj_concrete_temp")
        self.obj_temp = build_scheduled_value(self.obj_temp, "obj_temp")

        self.yx_prior_mean = build_scheduled_value(self.yx_prior_mean, "yx_prior_mean")
        self.yx_prior_std = build_scheduled_value(self.yx_prior_std, "yx_prior_std")

        self.hw_prior_mean = build_scheduled_value(self.hw_prior_mean, "hw_prior_mean")
        self.hw_prior_std = build_scheduled_value(self.hw_prior_std, "hw_prior_std")

        self.training_wheels = build_scheduled_value(self.training_wheels, "training_wheels")

        self.anchor_boxes = np.array(self.anchor_boxes)

    def _independent_prior(self):
        return dict(
            cell_y_logit=Normal(loc=self.yx_prior_mean, scale=self.yx_prior_std),
            cell_x_logit=Normal(loc=self.yx_prior_mean, scale=self.yx_prior_std),
            height_logit=Normal(loc=self.hw_prior_mean, scale=self.hw_prior_std),
            width_logit=Normal(loc=self.hw_prior_mean, scale=self.hw_prior_std),
            attr=Normal(loc=self.attr_prior_mean, scale=self.attr_prior_std),
            z_logit=Normal(loc=self.z_prior_mean, scale=self.z_prior_std),
        )

    def _compute_kl(self, tensors, prior):
        # --- box ---

        cell_y_kl = tensors["cell_y_logit_dist"].kl_divergence(prior["cell_y_logit"])
        cell_x_kl = tensors["cell_x_logit_dist"].kl_divergence(prior["cell_x_logit"])
        height_kl = tensors["height_logit_dist"].kl_divergence(prior["height_logit"])
        width_kl = tensors["width_logit_dist"].kl_divergence(prior["width_logit"])

        if "cell_y" in self.no_gradient:
            cell_y_kl = tf.stop_gradient(cell_y_kl)

        if "cell_x" in self.no_gradient:
            cell_x_kl = tf.stop_gradient(cell_x_kl)

        if "height" in self.no_gradient:
            height_kl = tf.stop_gradient(height_kl)

        if "width" in self.no_gradient:
            width_kl = tf.stop_gradient(width_kl)

        box_kl = tf.concat([cell_y_kl, cell_x_kl, height_kl, width_kl], axis=-1)

        # --- attr ---

        attr_kl = tensors["attr_dist"].kl_divergence(prior["attr"])

        if "attr" in self.no_gradient:
            attr_kl = tf.stop_gradient(attr_kl)

        # --- z ---

        z_kl = tensors["z_logit_dist"].kl_divergence(prior["z_logit"])

        if "z" in self.no_gradient:
            z_kl = tf.stop_gradient(z_kl)

        if "z" in self.fixed_values:
            z_kl = tf.zeros_like(z_kl)

        # --- obj ---

        # TODO: allow a choice about which prior to use in _compute_obj_kl

        obj_kl_tensors = self._compute_obj_kl(tensors)

        return dict(
            cell_y_kl=cell_y_kl,
            cell_x_kl=cell_x_kl,
            height_kl=height_kl,
            width_kl=width_kl,
            box_kl=box_kl,
            z_kl=z_kl,
            attr_kl=attr_kl,
            **obj_kl_tensors,
        )

    def _build_box(self, box_params, h, w, b, is_training):
        mean, log_std = tf.split(box_params, 2, axis=-1)

        std = self.std_nonlinearity(log_std)

        mean = self.training_wheels * tf.stop_gradient(mean) + (1-self.training_wheels) * mean
        std = self.training_wheels * tf.stop_gradient(std) + (1-self.training_wheels) * std

        cy_mean, cx_mean, height_mean, width_mean = tf.split(mean, 4, axis=-1)
        cy_std, cx_std, height_std, width_std = tf.split(std, 4, axis=-1)

        cy_logit_dist = Normal(loc=cy_mean, scale=cy_std)
        cy_logits = cy_logit_dist.sample()

        cx_logit_dist = Normal(loc=cx_mean, scale=cx_std)
        cx_logits = cx_logit_dist.sample()

        height_logit_dist = Normal(loc=height_mean, scale=height_std)
        height_logits = height_logit_dist.sample()

        width_logit_dist = Normal(loc=width_mean, scale=width_std)
        width_logits = width_logit_dist.sample()

        # --- cell y/x transform ---

        cell_y = tf.nn.sigmoid(tf.clip_by_value(cy_logits, -10, 10))
        cell_x = tf.nn.sigmoid(tf.clip_by_value(cx_logits, -10, 10))

        assert self.max_yx > self.min_yx

        cell_y = float(self.max_yx - self.min_yx) * cell_y + self.min_yx
        cell_x = float(self.max_yx - self.min_yx) * cell_x + self.min_yx

        # --- height/width transform ---

        height = tf.nn.sigmoid(tf.clip_by_value(height_logits, -10, 10))
        width = tf.nn.sigmoid(tf.clip_by_value(width_logits, -10, 10))
        assert self.max_hw > self.min_hw

        height = float(self.max_hw - self.min_hw) * height + self.min_hw
        width = float(self.max_hw - self.min_hw) * width + self.min_hw

        if "cell_y" in self.no_gradient:
            cell_y = tf.stop_gradient(cell_y)

        if "cell_x" in self.no_gradient:
            cell_x = tf.stop_gradient(cell_x)

        if "height" in self.no_gradient:
            height = tf.stop_gradient(height)

        if "width" in self.no_gradient:
            width = tf.stop_gradient(width)

        box = tf.concat([cell_y, cell_x, height, width], axis=-1)

        # --- Compute image-normalized box parameters ---

        # box height and width normalized to image height and width
        ys = height * self.anchor_boxes[b, 0] / self.image_height
        xs = width * self.anchor_boxes[b, 1] / self.image_width

        # box centre normalized to image height and width
        yt = (self.pixels_per_cell[0] / self.image_height) * (cell_y + h)
        xt = (self.pixels_per_cell[1] / self.image_width) * (cell_x + w)

        yt -= ys / 2
        xt -= xs / 2

        normalized_box = tf.concat([yt, xt, ys, xs], axis=-1)

        return dict(
            # "raw" box values
            cell_y=cell_y,
            cell_x=cell_x,
            height=height,
            width=width,
            box=box,

            cell_y_logit_dist=cy_logit_dist,
            cell_x_logit_dist=cx_logit_dist,
            height_logit_dist=height_logit_dist,
            width_logit_dist=width_logit_dist,

            # box top/left and height/width, in a coordinate frame where (0, 0) is image top-left
            # and (1, 1) is image bottom-right
            yt=yt,
            xt=xt,
            ys=ys,
            xs=xs,
            normalized_box=normalized_box
        )

    def _build_attr_from_image(self, inp, normalized_box, is_training):
        # --- Get object attributes using object encoder ---

        yt, xt, ys, xs = tf.split(normalized_box, 4, axis=-1)

        # yt/xt give top/left but here we need center
        yt += ys / 2
        xt += xs / 2

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)

        _boxes = tf.concat([xs, 2*xt - 1, ys, 2*yt - 1], axis=-1)

        grid_coords = warper(_boxes)
        grid_coords = tf.reshape(grid_coords, (self.batch_size, 1, *self.object_shape, 2,))
        input_glimpses = resampler_edge.resampler_edge(inp, grid_coords)
        input_glimpses = tf.reshape(input_glimpses, (-1, *self.object_shape, self.image_depth))

        attr = self.object_encoder(input_glimpses, (1, 1, 2*self.A), self.is_training)
        attr = tf.reshape(attr, (-1, 2*self.A))

        return input_glimpses, attr

    def _build_obj(self, obj_logits, is_training, **kwargs):
        obj_logits = self.training_wheels * tf.stop_gradient(obj_logits) + (1-self.training_wheels) * obj_logits
        obj_logits = obj_logits / self.obj_temp

        obj_log_odds = tf.clip_by_value(obj_logits, -10., 10.)

        obj_pre_sigmoid = concrete_binary_pre_sigmoid_sample(
            obj_log_odds, self.obj_concrete_temp
        )
        raw_obj = tf.nn.sigmoid(obj_pre_sigmoid)

        if self.noisy:
            obj = (
                self.float_is_training * raw_obj
                + (1 - self.float_is_training) * tf.round(raw_obj)
            )
        else:
            obj = tf.round(raw_obj)

        if "obj" in self.no_gradient:
            obj = tf.stop_gradient(obj)

        if "obj" in self.fixed_values:
            obj = self.fixed_values['obj'] * tf.ones_like(obj)

        return dict(
            obj=obj,
            raw_obj=raw_obj,
            obj_pre_sigmoid=obj_pre_sigmoid,
            obj_log_odds=obj_log_odds,
            obj_prob=tf.nn.sigmoid(obj_log_odds),
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

    def _make_empty(self):
        return np.array([{} for i in range(self.H * self.W * self.B)]).reshape(self.H, self.W, self.B)

    def _call(self, inp, inp_features, background, is_training):

        # --- set up sub networks and attributes ---

        self.maybe_build_subnet("box_network", builder=cfg.build_lateral, key="box")
        self.maybe_build_subnet("z_network", builder=cfg.build_lateral, key="z")
        self.maybe_build_subnet("obj_network", builder=cfg.build_lateral, key="obj")

        self.maybe_build_subnet("object_encoder")
        self.maybe_build_subnet("object_decoder")

        _, H, W, _, _ = inp_features.shape
        H = int(H)
        W = int(W)

        if not self.initialized:
            # Note this limits the re-usability of this module to images
            # with a fixed shape (the shape of the first image it is used on)
            self.image_height = int(inp.shape[-3])
            self.image_width = int(inp.shape[-2])
            self.image_depth = int(inp.shape[-1])
            self.H = H
            self.W = W
            self.HWB = H*W*self.B
            self.batch_size = tf.shape(inp)[0]
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

        _tensors = collections.defaultdict(self._make_empty)
        program = np.empty((H, W, self.B), dtype=np.object)

        # --- build the program ---

        for h, w, b in itertools.product(range(H), range(W), range(self.B)):
            partial_program, features = None, None
            _inp_features = inp_features[:, h, w, b, :]
            context = self._get_sequential_context(program, h, w, b, edge_element)

            # --- box ---

            layer_inp = tf.concat([_inp_features, context], axis=1)
            n_features = self.n_passthrough_features
            output_size = 8

            network_output = self.box_network(layer_inp, output_size + n_features, self. is_training)
            rep_input, features = tf.split(network_output, (output_size, n_features), axis=1)

            built = self._build_box(rep_input, h, w, b, self.is_training)

            for key, value in built.items():
                _tensors[key][h, w, b] = value
            partial_program = built['box']

            # --- attr ---

            input_glimpses, attr = self._build_attr_from_image(inp, built['normalized_box'], self.is_training)

            attr_mean, attr_log_std = tf.split(attr, [self.A, self.A], axis=-1)
            attr_std = self.std_nonlinearity(attr_log_std)

            attr_dist = Normal(loc=attr_mean, scale=attr_std)
            attr = attr_dist.sample()

            if "attr" in self.no_gradient:
                attr = tf.stop_gradient(attr)

            built = dict(attr_dist=attr_dist, attr=attr, input_glimpses=input_glimpses)

            for key, value in built.items():
                _tensors[key][h, w, b] = value
            partial_program = tf.concat([partial_program, built['attr']], axis=1)

            # --- z ---

            layer_inp = tf.concat([_inp_features, context, features, partial_program], axis=1)
            n_features = self.n_passthrough_features

            network_output = self.z_network(layer_inp, 2 + n_features, self.is_training)
            z_mean, z_log_std, features = tf.split(network_output, (1, 1, n_features), axis=1)
            z_std = self.std_nonlinearity(z_log_std)

            z_mean = self.training_wheels * tf.stop_gradient(z_mean) + (1-self.training_wheels) * z_mean
            z_std = self.training_wheels * tf.stop_gradient(z_std) + (1-self.training_wheels) * z_std
            z_logit_dist = Normal(loc=z_mean, scale=z_std)
            z_logits = z_logit_dist.sample()
            z = self.z_nonlinearity(z_logits)

            if "z" in self.no_gradient:
                z = tf.stop_gradient(z)

            if "z" in self.fixed_values:
                z = self.fixed_values['z'] * tf.ones_like(z)

            built = dict(z_logit_dist=z_logit_dist, z=z)

            for key, value in built.items():
                _tensors[key][h, w, b] = value
            partial_program = tf.concat([partial_program, built['z']], axis=1)

            # --- obj ---

            layer_inp = tf.concat([_inp_features, context, features, partial_program], axis=1)
            rep_input = self.obj_network(layer_inp, 1, self.is_training)

            built = self._build_obj(rep_input, self.is_training)

            for key, value in built.items():
                _tensors[key][h, w, b] = value
            partial_program = tf.concat([partial_program, built['obj']], axis=1)

            # --- final ---

            program[h, w, b] = partial_program
            assert program[h, w, b].shape[1] == total_sample_size

        # --- merge tensors from different grid cells ---

        tensors = dict(background=background)
        for k, v in _tensors.items():
            if k.endswith('_dist'):
                dist = v[0, 0, 0]
                dist_class = type(dist)
                params = dist.parameters.copy()
                tensor_keys = sorted(key for key, t in params.items() if isinstance(t, tf.Tensor))
                tensor_params = {}

                for key in tensor_keys:
                    t1 = []
                    for h in range(H):
                        t2 = []
                        for w in range(W):
                            t2.append(tf.stack([v[h, w, b].parameters[key] for b in range(self.B)], axis=1))
                        t1.append(tf.stack(t2, axis=1))
                    tensor_params[key] = tf.stack(t1, axis=1)

                params.update(tensor_params)
                tensors[k] = dist_class(**params)
            else:
                t1 = []
                for h in range(H):
                    t2 = []
                    for w in range(W):
                        t2.append(tf.stack([v[h, w, b] for b in range(self.B)], axis=1))
                    t1.append(tf.stack(t2, axis=1))
                tensors[k] = tf.stack(t1, axis=1)

        tensors["all"] = tf.concat(
            [tensors["box"], tensors["attr"], tensors["z"], tensors["obj"]], axis=-1)

        # --- kl ---

        prior = self._independent_prior()
        kl_tensors = self._compute_kl(tensors, prior)
        tensors.update(kl_tensors)

        tensors["n_objects"] = tf.fill((self.batch_size,), self.HWB)
        tensors["pred_n_objects"] = tf.reduce_sum(tensors['obj'], axis=(1, 2, 3, 4))
        tensors["pred_n_objects_hard"] = tf.reduce_sum(tf.round(tensors['obj']), axis=(1, 2, 3, 4))

        # --- compute sprite appearances from attr using object decoder ---

        object_decoder_in = tf.reshape(tensors["attr"], (self.batch_size * self.HWB, 1, 1, self.A))

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth+1,), self.is_training)

        object_logits = object_logits * ([self.obj_logit_scale] * 3 + [self.alpha_logit_scale])
        object_logits = object_logits + ([0.] * 3 + [self.alpha_logit_bias])

        objects = tf.nn.sigmoid(tf.clip_by_value(object_logits, -10., 10.))
        objects_shape = (self.batch_size, self.H, self.W, self.B, *self.object_shape, self.image_depth+1,)
        tensors["objects"] = tf.reshape(objects, objects_shape)

        # --- render ---

        render_tensors = self.render(tensors)
        tensors.update(render_tensors)

        return tensors

    def _compute_obj_kl(self, tensors):
        obj_kl_tensors = {}

        # --- compute obj_kl ---

        count_support = tf.range(self.HWB+1, dtype=tf.float32)

        if self.count_prior_dist is not None:
            if self.count_prior_dist is not None:
                assert len(self.count_prior_dist) == (self.HWB + 1)
            count_distribution = tf.constant(self.count_prior_dist, dtype=tf.float32)
        else:
            count_prior_prob = tf.nn.sigmoid(self.count_prior_log_odds)
            count_distribution = (1 - count_prior_prob) * (count_prior_prob ** count_support)

        normalizer = tf.reduce_sum(count_distribution)
        count_distribution = count_distribution / normalizer
        count_distribution = tf.tile(count_distribution[None, :], (self.batch_size, 1))
        count_so_far = tf.zeros((self.batch_size, 1), dtype=tf.float32)

        i = 0

        obj_kl = []

        for h, w, b in itertools.product(range(self.H), range(self.W), range(self.B)):
            p_z_given_Cz = tf.maximum(count_support[None, :] - count_so_far, 0) / (self.HWB - i)

            # Reshape for batch matmul
            _count_distribution = count_distribution[:, None, :]
            _p_z_given_Cz = p_z_given_Cz[:, :, None]

            p_z = tf.matmul(_count_distribution, _p_z_given_Cz)[:, :, 0]

            if self.use_concrete_kl:
                prior_log_odds = tf_safe_log(p_z) - tf_safe_log(1-p_z)
                _obj_kl = concrete_binary_sample_kl(
                    tensors["obj_pre_sigmoid"][:, h, w, b, :],
                    prior_log_odds, self.obj_concrete_temp,
                    tensors["obj_log_odds"][:, h, w, b, :],
                    self.obj_concrete_temp
                )
            else:
                prob = tensors["obj_prob"][:, h, w, b, :]

                _obj_kl = (
                    prob * (tf_safe_log(prob) - tf_safe_log(p_z))
                    + (1-prob) * (tf_safe_log(1-prob) - tf_safe_log(1-p_z))
                )

            obj_kl.append(_obj_kl)

            sample = tf.to_float(tensors["obj"][:, h, w, b, :] > 0.5)
            mult = sample * p_z_given_Cz + (1-sample) * (1-p_z_given_Cz)
            count_distribution = mult * count_distribution
            normalizer = tf.reduce_sum(count_distribution, axis=1, keepdims=True)
            normalizer = tf.maximum(normalizer, 1e-6)
            count_distribution = count_distribution / normalizer

            count_so_far += sample

            i += 1

        if "obj" in self.no_gradient:
            obj_kl = tf.stop_gradient(obj_kl)

        obj_kl_tensors["obj_kl"] = tf.reshape(
            tf.concat(obj_kl, axis=1),
            (self.batch_size, self.H, self.W, self.B, 1))

        return obj_kl_tensors

    def render(self, tensors):
        render_tensors = {}

        # --- Compute sprite locations from box parameters ---

        objects = tf.reshape(
            tensors["objects"],
            (self.batch_size, self.HWB, *self.object_shape, self.image_depth+1,))

        obj_img, obj_alpha = tf.split(objects, [3, 1], axis=-1)

        if "alpha" in self.no_gradient:
            obj_alpha = tf.stop_gradient(obj_alpha)

        if "alpha" in self.fixed_values:
            obj_alpha = float(self.fixed_values["alpha"]) * tf.ones_like(obj_alpha)

        obj_alpha *= tf.reshape(tensors['obj'], (self.batch_size, self.HWB, 1, 1, 1))

        z = tf.reshape(tensors['z'], (self.batch_size, self.HWB, 1, 1, 1))
        obj_importance = tf.maximum(obj_alpha * z, 0.01)

        objects = tf.concat([obj_img, obj_alpha, obj_importance], axis=-1)

        # --- Compose images ---

        ys, xs, yt, xt = tensors["ys"], tensors["xs"], tensors["yt"], tensors["xt"]

        scales = tf.concat([ys, xs], axis=-1)
        scales = tf.reshape(scales, (self.batch_size, self.HWB, 2))

        offsets = tf.concat([yt, xt], axis=-1)
        offsets = tf.reshape(offsets, (self.batch_size, self.HWB, 2))

        output = render_sprites.render_sprites(
            objects,
            tensors["n_objects"],
            scales,
            offsets,
            tensors["background"]
        )

        # --- Store values ---

        render_tensors['area'] = (ys * float(self.image_height)) * (xs * float(self.image_width))
        render_tensors['output'] = output

        return render_tensors
