import tensorflow as tf
import numpy as np
import collections
import sonnet as snt

from dps import cfg
from dps.utils import Param
from dps.utils.tf import build_scheduled_value, FIXED_COLLECTION, ScopedFunction

from auto_yolo.tf_ops import render_sprites, resampler_edge
from auto_yolo.models.core import (
    normal_vae, concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl, tf_safe_log)


class ObjectLayer(ScopedFunction):
    object_shape = Param()
    anchor_boxes = Param()

    attr_prior_mean = Param()
    attr_prior_std = Param()
    A = Param()
    noisy = Param()

    min_hw = Param()
    max_hw = Param()

    min_yx = Param()
    max_yx = Param()

    n_passthrough_features = Param()

    use_concrete_kl = Param()
    count_prior_log_odds = Param()
    count_prior_dist = Param()
    obj_concrete_temp = Param(help="Higher values -> smoother")
    obj_temp = Param(help="Higher values -> more uniform")

    yx_prior_mean = Param()
    yx_prior_std = Param()

    hw_prior_mean = Param()
    hw_prior_std = Param()

    z_prior_mean = Param()
    z_prior_std = Param()

    obj_logit_scale = Param()
    alpha_logit_scale = Param()
    alpha_logit_bias = Param()

    training_wheels = Param()
    n_lookback = Param()

    box_network = None
    z_network = None
    obj_network = None
    object_encoder = None
    object_decoder = None
    edge_weights = None

    def __init__(self, pixels_per_cell, scope=None, **kwargs):
        super(ObjectLayer, self).__init__(scope=scope, **kwargs)

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

    @staticmethod
    def std_nonlinearity(std_logit):
        # return tf.exp(std)
        return 2 * tf.nn.sigmoid(tf.clip_by_value(std_logit, -10, 10))

    def _build_box(self, box_params, is_training):
        mean, log_std = tf.split(box_params, 2, axis=-1)

        std = self.std_nonlinearity(log_std)
        if not self.noisy:
            std = tf.zeros_like(std)

        mean = self.training_wheels * tf.stop_gradient(mean) + (1-self.training_wheels) * mean
        std = self.training_wheels * tf.stop_gradient(std) + (1-self.training_wheels) * std

        cy_mean, cx_mean, h_mean, w_mean = tf.split(mean, 4, axis=-1)
        cy_std, cx_std, h_std, w_std = tf.split(std, 4, axis=-1)

        cy_logits, cy_kl = normal_vae(cy_mean, cy_std, self.yx_prior_mean, self.yx_prior_std)
        cx_logits, cx_kl = normal_vae(cx_mean, cx_std, self.yx_prior_mean, self.yx_prior_std)

        h_logits, h_kl = normal_vae(h_mean, h_std, self.hw_prior_mean, self.hw_prior_std)
        w_logits, w_kl = normal_vae(w_mean, w_std, self.hw_prior_mean, self.hw_prior_std)

        cell_y = tf.nn.sigmoid(tf.clip_by_value(cy_logits, -10, 10))
        cell_x = tf.nn.sigmoid(tf.clip_by_value(cx_logits, -10, 10))
        h = tf.nn.sigmoid(tf.clip_by_value(h_logits, -10, 10))
        w = tf.nn.sigmoid(tf.clip_by_value(w_logits, -10, 10))

        assert self.max_yx > self.min_yx

        cell_y = float(self.max_yx - self.min_yx) * cell_y + self.min_yx
        cell_x = float(self.max_yx - self.min_yx) * cell_x + self.min_yx

        assert self.max_hw > self.min_hw

        h = float(self.max_hw - self.min_hw) * h + self.min_hw
        w = float(self.max_hw - self.min_hw) * w + self.min_hw

        if "cell_y" in self.no_gradient:
            cell_y = tf.stop_gradient(cell_y)
            cy_kl = tf.stop_gradient(cy_kl)

        if "cell_x" in self.no_gradient:
            cell_x = tf.stop_gradient(cell_x)
            cx_kl = tf.stop_gradient(cx_kl)

        if "h" in self.no_gradient:
            h = tf.stop_gradient(h)
            h_kl = tf.stop_gradient(h_kl)

        if "w" in self.no_gradient:
            w = tf.stop_gradient(w)
            w_kl = tf.stop_gradient(w_kl)

        box = tf.concat([cell_y, cell_x, h, w], axis=-1)
        box_kl = tf.concat([cy_kl, cx_kl, h_kl, w_kl], axis=-1)

        return dict(
            cell_y_mean=cy_mean,
            cell_x_mean=cx_mean,
            h_mean=h_mean,
            w_mean=w_mean,

            cell_y_std=cy_std,
            cell_x_std=cx_std,
            h_std=h_std,
            w_std=w_std,

            cell_y=cell_y,
            cell_x=cell_x,
            h=h,
            w=w,

            cell_y_kl=cy_kl,
            cell_x_kl=cx_kl,
            h_kl=h_kl,
            w_kl=w_kl,

            box_kl=box_kl,
            box=box,
        )

    def _build_attr_from_image(self, inp, boxes, h, w, b, is_training):

        # --- Compute sprite locations from box parameters ---

        cell_y, cell_x, height, width = tf.split(boxes, 4, axis=-1)

        # box height and width normalized to image height and width
        ys = height * self.anchor_boxes[b, 0] / self.image_height
        xs = width * self.anchor_boxes[b, 1] / self.image_width

        # box centre normalized to image height and width
        yt = (self.pixels_per_cell[0] / self.image_height) * (cell_y + h)
        xt = (self.pixels_per_cell[1] / self.image_width) * (cell_x + w)

        # --- Get object attributes using object encoder ---

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

        # --- set up sub networks ---

        if self.box_network is None:
            self.box_network = cfg.build_lateral(scope="box_lateral_network")
            if "box" in self.fixed_weights:
                self.box_network.fix_variables()

        if self.object_encoder is None:
            self.object_encoder = cfg.build_object_encoder(scope="object_encoder")
            if "object_encoder" in self.fixed_weights:
                self.object_encoder.fix_variables()

        if self.object_decoder is None:
            self.object_decoder = cfg.build_object_decoder(scope="object_decoder")
            if "object_decoder" in self.fixed_weights:
                self.object_decoder.fix_variables()

        if self.z_network is None:
            self.z_network = cfg.build_lateral(scope="z_lateral_network")
            if "z" in self.fixed_weights:
                self.z_network.fix_variables()

        if self.obj_network is None:
            self.obj_network = cfg.build_lateral(scope="obj_lateral_network")
            if "obj" in self.fixed_weights:
                self.obj_network.fix_variables()

        _, H, W, _, _ = inp_features.shape
        H = int(H)
        W = int(W)

        if not self.initialized:
            # Note this limits the re-usability of this module within
            self.image_height = int(inp.shape[-3])
            self.image_width = int(inp.shape[-2])
            self.image_depth = int(inp.shape[-1])
            self.H = H
            self.W = W
            self.HWB = H*W*self.B
            self.batch_size = tf.shape(inp)[0]
            self.is_training = is_training
            self.float_is_training = tf.to_float(is_training)

        # --- set-up the edge element ---

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

        for h in range(self.H):
            for w in range(self.W):
                for b in range(self.B):
                    partial_program, features = None, None
                    _inp_features = inp_features[:, h, w, b, :]
                    context = self._get_sequential_context(program, h, w, b, edge_element)

                    # --- box ---

                    layer_inp = tf.concat([_inp_features, context], axis=1)
                    n_features = self.n_passthrough_features
                    output_size = 8

                    network_output = self.box_network(layer_inp, output_size + n_features, self. is_training)
                    rep_input, features = tf.split(network_output, (output_size, n_features), axis=1)

                    built = self._build_box(rep_input, self.is_training)

                    for key, value in built.items():
                        _tensors[key][h, w, b] = value
                    partial_program = built['box']

                    # --- attr ---

                    input_glimpses, attr = self._build_attr_from_image(inp, built['box'], h, w, b, self.is_training)

                    attr_mean, attr_log_std = tf.split(attr, [self.A, self.A], axis=-1)
                    attr_std = self.std_nonlinearity(attr_log_std)

                    if not self.noisy:
                        attr_std = tf.zeros_like(attr_std)

                    attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

                    if "attr" in self.no_gradient:
                        attr = tf.stop_gradient(attr)
                        attr_kl = tf.stop_gradient(attr_kl)

                    built = dict(attr_mean=attr_mean, attr_std=attr_std, attr=attr,
                                 attr_kl=attr_kl, input_glimpses=input_glimpses)

                    for key, value in built.items():
                        _tensors[key][h, w, b] = value
                    partial_program = tf.concat([partial_program, built['attr']], axis=1)

                    # --- z ---

                    layer_inp = tf.concat([_inp_features, context, features, partial_program], axis=1)
                    n_features = self.n_passthrough_features

                    network_output = self.z_network(layer_inp, 2 + n_features, self.is_training)
                    z_mean, z_log_std, features = tf.split(network_output, (1, 1, n_features), axis=1)
                    z_std = self.std_nonlinearity(z_log_std)
                    if not self.noisy:
                        z_std = tf.zeros_like(z_std)

                    z_mean = self.training_wheels * tf.stop_gradient(z_mean) + (1-self.training_wheels) * z_mean
                    z_std = self.training_wheels * tf.stop_gradient(z_std) + (1-self.training_wheels) * z_std
                    z_logits, z_kl = normal_vae(z_mean, z_std, self.z_prior_mean, self.z_prior_std)
                    z = 4 * tf.nn.sigmoid(tf.clip_by_value(z_logits, -10, 10))

                    if "z" in self.no_gradient:
                        z = tf.stop_gradient(z)
                        z_kl = tf.stop_gradient(z_kl)

                    if "z" in self.fixed_values:
                        z = self.fixed_values['z'] * tf.ones_like(z)
                        z_kl = tf.zeros_like(z_kl)

                    built = dict(z_mean=z_mean, z_std=z_std, z=z, z_kl=z_kl,)

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

        tensors = dict(background=background)
        for k, v in _tensors.items():
            t1 = []
            for h in range(H):
                t2 = []
                for w in range(W):
                    t2.append(tf.stack([v[h, w, b] for b in range(self.B)], axis=1))
                t1.append(tf.stack(t2, axis=1))
            tensors[k] = tf.stack(t1, axis=1)

        tensors["all"] = tf.concat(
            [tensors["box"], tensors["attr"], tensors["z"], tensors["obj"]], axis=-1)

        # --- get obj kl ---

        obj_kl_tensors = self.obj_kl(tensors)
        tensors.update(obj_kl_tensors)

        tensors["n_objects"] = tf.fill((self.batch_size,), self.HWB)
        tensors["pred_n_objects"] = tf.reduce_sum(tensors['obj'], axis=(1, 2, 3, 4))
        tensors["pred_n_objects_hard"] = tf.reduce_sum(tf.round(tensors['obj']), axis=(1, 2, 3, 4))

        # --- Compute sprite appearances from attr using object decoder ---

        object_decoder_in = tf.reshape(tensors["attr"], (self.batch_size * self.HWB, 1, 1, self.A))

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth+1,), self.is_training)

        object_logits = object_logits * ([self.obj_logit_scale] * 3 + [self.alpha_logit_scale])
        object_logits = object_logits + ([0.] * 3 + [self.alpha_logit_bias])

        objects = tf.nn.sigmoid(tf.clip_by_value(object_logits, -10., 10.))
        objects_shape = (self.batch_size, self.H, self.W, self.B,
                         *self.object_shape, self.image_depth+1,)
        tensors["objects"] = tf.reshape(objects, objects_shape)

        # --- render ---

        render_tensors = self.render(tensors)
        tensors.update(render_tensors)

        return tensors

    def obj_kl(self, tensors):
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

        for h in range(self.H):
            for w in range(self.W):
                for b in range(self.B):
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

        # All in cell-local co-ordinates, should be invariant to image size.
        cell_y, cell_x, h, w = tf.split(tensors['box'], 4, axis=-1)

        anchor_box_h = self.anchor_boxes[:, 0].reshape(1, 1, 1, self.B, 1)
        anchor_box_w = self.anchor_boxes[:, 1].reshape(1, 1, 1, self.B, 1)

        # box height and width normalized to image height and width
        ys = h * anchor_box_h / self.image_height
        xs = w * anchor_box_w / self.image_width

        # box centre normalized to image height and width
        yt = (
            (self.pixels_per_cell[0] / self.image_height)
            * (cell_y + tf.range(self.H, dtype=tf.float32)[None, :, None, None, None])
        )
        xt = (
            (self.pixels_per_cell[1] / self.image_width)
            * (cell_x + tf.range(self.W, dtype=tf.float32)[None, None, :, None, None])
        )

        # `render_sprites` requires box top-left, whereas y and x give box center
        yt -= ys / 2
        xt -= xs / 2

        render_tensors["normalized_box"] = tf.concat([yt, xt, ys, xs], axis=-1)

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

        render_tensors['latent_hw'] = tensors['box'][..., 2:]
        render_tensors['latent_area'] = h * w
        render_tensors['area'] = (ys * float(self.image_height)) * (xs * float(self.image_width))
        render_tensors['output'] = output

        return render_tensors
