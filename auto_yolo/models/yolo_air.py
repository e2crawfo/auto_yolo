import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb
import collections
import sonnet as snt
import os
import shutil

from dps import cfg
from dps.utils import Param
from dps.utils.tf import ScopedFunction, tf_mean_sum, build_scheduled_value, FIXED_COLLECTION

from auto_yolo.tf_ops import render_sprites
from auto_yolo.models.core import (
    AP, loss_builders, normal_vae, concrete_binary_pre_sigmoid_sample, concrete_binary_sample_kl)


def tf_safe_log(value, nan_value=100.0):
    log_value = tf.log(value)
    log_value = tf.where(tf.is_nan(log_value), -100.0 * tf.ones_like(log_value), log_value)
    return log_value


class YoloAir_Network(ScopedFunction):
    pixels_per_cell = Param()
    object_shape = Param()
    anchor_boxes = Param(help="List of (h, w) pairs.")

    A = Param(100, help="Dimension of attribute vector.")

    min_hw = Param(0.0)
    max_hw = Param(1.0)

    min_yx = Param(0.0)
    max_yx = Param(1.0)

    n_backbone_features = Param(100)
    n_passthrough_features = Param(100)

    xent_loss = Param(True)

    fixed_values = Param(dict())
    fixed_weights = Param("")
    no_gradient = Param("")

    use_concrete_kl = Param(True)
    count_prior_log_odds = Param()
    count_prior_dist = Param(None, help="If not None, overrides `count_prior_log_odds`.")
    obj_concrete_temp = Param(1.0, help="Higher values -> smoother")
    obj_temp = Param(1.0, help="Higher values -> more uniform")

    train_reconstruction = Param(True)
    train_kl = Param(True)
    noisy = Param(True)

    reconstruction_weight = Param(1.0)
    kl_weight = Param(1.0)

    yx_prior_mean = Param(0.0)
    yx_prior_std = Param(1.0)

    hw_prior_mean = Param(0.0)
    hw_prior_std = Param(1.0)

    attr_prior_mean = Param(0.0)
    attr_prior_std = Param(1.0)

    obj_logit_scale = Param(2.0)
    alpha_logit_scale = Param(0.1)
    alpha_logit_bias = Param(5.0)

    training_wheels = Param(0.0)

    sequential_cfg = Param(dict(
        on=False,
        n_lookback=1,
    ))
    incremental_attr = Param(True)
    attr_context = Param(False)

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.H = int(np.ceil(self.image_height / self.pixels_per_cell[0]))
        self.W = int(np.ceil(self.image_width / self.pixels_per_cell[1]))
        self.B = len(self.anchor_boxes)
        self.HWB = self.H * self.W * self.B

        if isinstance(self.count_prior_dist, str):
            self.count_prior_dist = eval(self.count_prior_dist)

        if self.count_prior_dist is not None:
            assert len(self.count_prior_dist) == (self.HWB + 1)

        self.count_prior_log_odds = build_scheduled_value(self.count_prior_log_odds, "count_prior_log_odds")
        self.obj_concrete_temp = build_scheduled_value(self.obj_concrete_temp, "obj_concrete_temp")
        self.obj_temp = build_scheduled_value(self.obj_temp, "obj_temp")

        self.yx_prior_mean = build_scheduled_value(self.yx_prior_mean, "yx_prior_mean")
        self.yx_prior_std = build_scheduled_value(self.yx_prior_std, "yx_prior_std")

        self.hw_prior_mean = build_scheduled_value(self.hw_prior_mean, "hw_prior_mean")
        self.hw_prior_std = build_scheduled_value(self.hw_prior_std, "hw_prior_std")

        self.attr_prior_mean = build_scheduled_value(self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(self.attr_prior_std, "attr_prior_std")

        self.training_wheels = build_scheduled_value(self.training_wheels, "training_wheels")

        self.reconstruction_weight = build_scheduled_value(self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")

        if not self.noisy and self.train_kl:
            raise Exception("If `noisy` is False, `train_kl` must also be False.")

        self.anchor_boxes = np.array(self.anchor_boxes)

        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = AP(ap_iou_values)

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        self.backbone = None
        self.box_network = None
        self.attr_network = None
        self.obj_network = None
        self.object_encoder = None
        self.object_decoder = None

        super(YoloAir_Network, self).__init__(scope=scope)

    @property
    def inp(self):
        return self._tensors["inp"]

    @property
    def batch_size(self):
        return self._tensors["batch_size"]

    @property
    def is_training(self):
        return self._tensors["is_training"]

    @property
    def float_is_training(self):
        return self._tensors["float_is_training"]

    def _get_scheduled_value(self, name):
        scalar = self._tensors.get(name, None)
        if scalar is None:
            schedule = getattr(self, name)
            scalar = self._tensors[name] = build_scheduled_value(schedule, name)
        return scalar

    def _build_box(self, box_params, is_training):
        mean, log_std = tf.split(box_params, 2, axis=-1)
        std = tf.exp(log_std)
        if not self.noisy:
            std = tf.zeros_like(log_std)

        cy_mean, cx_mean, h_mean, w_mean = tf.split(mean, 4, axis=-1)
        cy_std, cx_std, h_std, w_std = tf.split(std, 4, axis=-1)

        cy_logits, cy_kl = normal_vae(cy_mean, cy_std, self.yx_prior_mean, self.yx_prior_std)
        cx_logits, cx_kl = normal_vae(cx_mean, cx_std, self.yx_prior_mean, self.yx_prior_std)

        h_logits, h_kl = normal_vae(h_mean, h_std, self.hw_prior_mean, self.hw_prior_std)
        w_logits, w_kl = normal_vae(w_mean, w_std, self.hw_prior_mean, self.hw_prior_std)

        cell_y = tf.nn.sigmoid(tf.clip_by_value(cy_logits, -10, 10.))
        cell_x = tf.nn.sigmoid(tf.clip_by_value(cx_logits, -10, 10.))
        h = tf.nn.sigmoid(tf.clip_by_value(h_logits, -10, 10.))
        w = tf.nn.sigmoid(tf.clip_by_value(w_logits, -10, 10.))

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

    def _build_attr_from_image(self, boxes, h, w, b, is_training):

        # --- Compute sprite locations from box parameters ---

        cell_y, cell_x, height, width = tf.split(boxes, 4, axis=-1)

        # box height and width normalized to image height and width
        ys = height * self.anchor_boxes[b, 0] / self.image_height
        xs = width * self.anchor_boxes[b, 1] / self.image_width

        # box centre normalized to image height and width
        yt = (self.pixels_per_cell[0] / self.image_height) * (cell_y + h)
        xt = (self.pixels_per_cell[1] / self.image_width) * (cell_x + w)

        # `render_sprites` requires box top-left, whereas y and x give box center
        yt -= ys / 2
        xt -= xs / 2

        # --- Get object attributes using object encoder ---

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)

        _boxes = tf.concat([xs, 2*(xt + xs/2) - 1, ys, 2*(yt + ys/2) - 1], axis=-1)

        grid_coords = warper(_boxes)
        grid_coords = tf.reshape(grid_coords, (self.batch_size, 1, *self.object_shape, 2,))
        input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)
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
                self.float_is_training * raw_obj +
                (1 - self.float_is_training) * tf.round(raw_obj)
            )
        else:
            obj = tf.round(raw_obj)

        if "obj" in self.no_gradient:
            obj = tf.stop_gradient(obj)

        return dict(
            obj=obj,
            raw_obj=raw_obj,
            obj_pre_sigmoid=obj_pre_sigmoid,
            obj_log_odds=obj_log_odds,
            obj_prob=tf.nn.sigmoid(obj_log_odds),
        )

    def _get_sequential_input(self, program, h, w, b, edge_element):
        inp = []
        grid_size = 2 * self.sequential_cfg.n_lookback + 1
        n_grid_locs = int((grid_size**2) / 2)

        for idx in range(n_grid_locs):
            _i = int(idx / grid_size) + h - self.sequential_cfg.n_lookback
            _j = int(idx % grid_size) + w - self.sequential_cfg.n_lookback

            for k in range(self.B):
                if _i < 0 or _j < 0 or _i >= program.shape[0] or _j >= program.shape[1]:
                    inp.append(edge_element)
                else:
                    inp.append(program[_i, _j, k])

        offset = -(self.B - 1) + b
        for k in range(self.B-1):
            _k = k + offset
            if _k < 0:
                inp.append(edge_element)
            else:
                inp.append(program[h, w, _k])

        return tf.concat(inp, axis=1)

    def _make_empty(self):
        return np.array([{} for i in range(self.H * self.W * self.B)]).reshape(self.H, self.W, self.B)

    def _build_program_generator_sequential(self):
        H, W, B = self.H, self.W, self.B

        if self.backbone is None:
            self.backbone = cfg.build_backbone(scope="backbone")
            self.backbone.layout[-1]['filters'] = B * self.n_backbone_features

            if "backbone" in self.fixed_weights:
                self.backbone.fix_variables()

        inp = self._tensors["inp"]
        backbone_output = self.backbone(inp, (H, W, B*self.n_backbone_features), self.is_training)
        backbone_output = tf.reshape(backbone_output, (-1, H, W, B, self.n_backbone_features))

        # --- set-up the edge element ---

        sizes = [4, self.A, 1] if self.incremental_attr else [4, 1]
        sigmoids = [True, False, True] if self.incremental_attr else [True, True]
        total_sample_size = sum(sizes)

        self.edge_weights = tf.get_variable("edge_weights", shape=total_sample_size, dtype=tf.float32)

        if "backbone" in self.fixed_weights:
            tf.add_to_collection(FIXED_COLLECTION, self.edge_weights)

        _edge_weights = tf.split(self.edge_weights, sizes, axis=0)
        _edge_weights = [
            (tf.nn.sigmoid(ew) if sigmoid else ew)
            for ew, sigmoid in zip(_edge_weights, sigmoids)]
        edge_element = tf.concat(_edge_weights, axis=0)
        edge_element = tf.tile(edge_element[None, :], (self.batch_size, 1))

        # --- initialize containers for storing built program ---

        _tensors = collections.defaultdict(self._make_empty)
        program = np.empty((H, W, B), dtype=np.object)

        # --- build the program ---

        for h in range(self.H):
            for w in range(self.W):
                for b in range(self.B):
                    partial_program, features = None, None
                    _backbone_output = backbone_output[:, h, w, b, :]
                    context = self._get_sequential_input(program, h, w, b, edge_element)

                    # --- box ---

                    if self.box_network is None:
                        self.box_network = self.sequential_cfg.build_next_step(scope="box_sequential_network")
                        if "box" in self.fixed_weights:
                            self.box_network.fix_variables()

                    layer_inp = tf.concat([_backbone_output, context], axis=1)
                    n_features = self.n_passthrough_features
                    output_size = 8

                    network_output = self.box_network(layer_inp, output_size + n_features, self.is_training)
                    rep_input, features = tf.split(network_output, (output_size, n_features), axis=1)

                    built = self._build_box(rep_input, self.is_training)

                    for key, value in built.items():
                        _tensors[key][h, w, b] = value
                    partial_program = built['box']

                    # --- attr ---

                    if self.incremental_attr:
                        input_glimpses, attr = self._build_attr_from_image(built['box'], h, w, b, self.is_training)

                        if self.attr_context:
                            # Get attr by combining context with the output of the object encoder
                            if self.attr_network is None:
                                self.attr_network = self.sequential_cfg.build_next_step(scope="attr_sequential_network")
                                if "attr" in self.fixed_weights:
                                    self.attr_network.fix_variables()

                            layer_inp = tf.concat([_backbone_output, context, features, partial_program, attr], axis=1)
                            n_features = self.n_passthrough_features
                            output_size = 2 * self.A

                            network_output = self.attr_network(layer_inp, output_size + n_features, self.is_training)
                            attr, features = tf.split(network_output, (output_size, n_features), axis=1)

                        attr_mean, attr_log_std = tf.split(attr, [self.A, self.A], axis=-1)
                        attr_std = tf.exp(attr_log_std)

                        if not self.noisy:
                            attr_std = tf.zeros_like(attr_std)

                        attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

                        if "attr" in self.no_gradient:
                            attr = tf.stop_gradient(attr)
                            attr_kl = tf.stop_gradient(attr_kl)

                        built = dict(
                            attr_mean=attr_mean,
                            attr_std=attr_std,
                            attr=attr,
                            attr_kl=attr_kl,
                            input_glimpses=input_glimpses
                        )

                        for key, value in built.items():
                            _tensors[key][h, w, b] = value
                        partial_program = tf.concat([partial_program, built['attr']], axis=1)

                    # --- obj ---

                    if self.obj_network is None:
                        self.obj_network = self.sequential_cfg.build_next_step(scope="obj_sequential_network")
                        if "obj" in self.fixed_weights:
                            self.obj_network.fix_variables()

                    layer_inp = tf.concat([_backbone_output, context, features, partial_program], axis=1)
                    rep_input = self.obj_network(layer_inp, 1, self.is_training)

                    built = self._build_obj(rep_input, self.is_training)

                    for key, value in built.items():
                        _tensors[key][h, w, b] = value
                    partial_program = tf.concat([partial_program, built['obj']], axis=1)

                    program[h, w, b] = partial_program
                    assert program[h, w, b].shape[1] == total_sample_size

        def form_tensor(pieces):
            t1 = []
            for h in range(H):
                t2 = []
                for w in range(W):
                    t2.append(tf.stack([pieces[h, w, b] for b in range(B)], axis=1))
                t1.append(tf.stack(t2, axis=1))
            return tf.stack(t1, axis=1)

        for key, value in _tensors.items():
            self._tensors[key] = form_tensor(value)

    def _build_program_interpreter(self):

        # --- Compute sprite locations from box parameters ---

        # All in cell-local co-ordinates, should be invariant to image size.
        boxes = self._tensors['box']
        cell_y, cell_x, h, w = tf.split(boxes, 4, axis=-1)

        anchor_box_h = self.anchor_boxes[:, 0].reshape(1, 1, 1, self.B, 1)
        anchor_box_w = self.anchor_boxes[:, 1].reshape(1, 1, 1, self.B, 1)

        # box height and width normalized to image height and width
        ys = h * anchor_box_h / self.image_height
        xs = w * anchor_box_w / self.image_width

        # box centre normalized to image height and width
        yt = (
            (self.pixels_per_cell[0] / self.image_height) *
            (cell_y + tf.range(self.H, dtype=tf.float32)[None, :, None, None, None])
        )
        xt = (
            (self.pixels_per_cell[1] / self.image_width) *
            (cell_x + tf.range(self.W, dtype=tf.float32)[None, None, :, None, None])
        )

        # `render_sprites` requires box top-left, whereas y and x give box center
        yt -= ys / 2
        xt -= xs / 2

        self._tensors["normalized_box"] = tf.concat([yt, xt, ys, xs], axis=-1)

        if self.incremental_attr:
            attr = self._tensors["attr"]
        else:
            # --- Get object attributes using object encoder ---
            transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
            warper = snt.AffineGridWarper(
                (self.image_height, self.image_width), self.object_shape, transform_constraints)

            _boxes = tf.concat([xs, 2*(xt + xs/2) - 1, ys, 2*(yt + ys/2) - 1], axis=-1)
            _boxes = tf.reshape(_boxes, (self.batch_size * self.HWB, 4))
            grid_coords = warper(_boxes)
            grid_coords = tf.reshape(grid_coords, (self.batch_size, self.HWB, *self.object_shape, 2,))
            input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)

            self._tensors["input_glimpses"] = tf.reshape(
                input_glimpses, (self.batch_size, self.H, self.W, self.B, *self.object_shape, self.image_depth))

            object_encoder_in = tf.reshape(
                input_glimpses,
                (self.batch_size * self.HWB, *self.object_shape, self.image_depth))

            attr = self.object_encoder(object_encoder_in, (1, 1, 2*self.A), self.is_training)

            attr = tf.reshape(attr, (self.batch_size, self.H, self.W, self.B, 2*self.A))

            attr_mean, attr_log_std = tf.split(attr, [self.A, self.A], axis=-1)
            attr_std = tf.exp(attr_log_std)

            attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

            if "attr" in self.no_gradient:
                attr = tf.stop_gradient(attr)
                attr_kl = tf.stop_gradient(attr_kl)

            self._tensors["attr_mean"] = attr_mean
            self._tensors["attr_std"] = attr_std
            self._tensors["attr"] = attr
            self._tensors["attr_kl"] = attr_kl

        object_decoder_in = tf.reshape(attr, (self.batch_size * self.HWB, 1, 1, self.A))

        # --- Compute sprites from attr using object decoder ---

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth+1,), self.is_training)

        object_logits = object_logits * ([self.obj_logit_scale] * 3 + [self.alpha_logit_scale])
        object_logits = object_logits + ([0.] * 3 + [self.alpha_logit_bias])

        objects = tf.nn.sigmoid(tf.clip_by_value(object_logits, -10., 10.))

        self._tensors["objects"] = tf.reshape(
            objects, (self.batch_size, self.H, self.W, self.B, *self.object_shape, self.image_depth+1,))

        objects = tf.reshape(objects, (self.batch_size, self.HWB, *self.object_shape, self.image_depth+1,))

        obj_img, obj_alpha = tf.split(objects, [3, 1], axis=-1)

        if "alpha" in self.no_gradient:
            obj_alpha = tf.stop_gradient(obj_alpha)

        if "alpha" in self.fixed_values:
            obj_alpha = float(self.fixed_values["alpha"]) * tf.ones_like(obj_alpha)

        obj_alpha *= tf.reshape(self._tensors['obj'], (self.batch_size, self.HWB, 1, 1, 1))
        obj_alpha = tf.exp(obj_alpha * 5 + (1-obj_alpha) * -5)  # Inner expression is equivalent to 5 * (2*obj_alpha - 1)

        objects = tf.concat([obj_img, obj_alpha], axis=-1)

        # --- Compose images ---

        scales = tf.concat([ys, xs], axis=-1)
        scales = tf.reshape(scales, (self.batch_size, self.HWB, 2))

        offsets = tf.concat([yt, xt], axis=-1)
        offsets = tf.reshape(offsets, (self.batch_size, self.HWB, 2))

        output = render_sprites.render_sprites(
            objects,
            self._tensors["n_objects"],
            scales,
            offsets,
            self._tensors["background"]
        )

        # output = tf.clip_by_value(output, 1e-6, 1-1e-6)

        # --- Store values ---

        self._tensors['latent_hw'] = boxes[..., 2:]
        self._tensors['latent_area'] = h * w
        self._tensors['area'] = (ys * float(self.image_height)) * (xs * float(self.image_width))

        self._tensors['output'] = output

    def _process_labels(self, labels):
        self._tensors.update(
            annotations=labels[0],
            n_annotations=labels[1],
            targets=labels[2],
        )

    def _call(self, inp, _, is_training):
        inp, labels, background = inp
        return self.build_graph(inp, labels, background, is_training)

    def build_graph(self, inp, labels, background, is_training):
        # --- initialize containers for storing outputs ---

        self._tensors = dict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            background=background,
            batch_size=tf.shape(inp)[0],
        )

        self._process_labels(labels)

        # --- build graph ---

        if self.object_encoder is None:
            self.object_encoder = cfg.build_object_encoder(scope="object_encoder")
            if "encoder" in self.fixed_weights:
                self.object_encoder.fix_variables()

        if self.object_decoder is None:
            self.object_decoder = cfg.build_object_decoder(scope="object_decoder")
            if "decoder" in self.fixed_weights:
                self.object_decoder.fix_variables()

        self._build_program_generator_sequential()

        # --- compute obj_kl ---

        count_support = tf.range(self.HWB+1, dtype=tf.float32)

        if self.count_prior_dist is not None:
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
                    p_z_given_Cz = (count_support[None, :] - count_so_far) / (self.HWB - i)

                    # Reshape for batch matmul
                    _count_distribution = count_distribution[:, None, :]
                    _p_z_given_Cz = p_z_given_Cz[:, :, None]

                    p_z = tf.matmul(_count_distribution, _p_z_given_Cz)[:, :, 0]

                    if self.use_concrete_kl:
                        prior_log_odds = tf.log(p_z / (1-p_z))
                        _obj_kl = concrete_binary_sample_kl(
                            self._tensors["obj_pre_sigmoid"][:, h, w, b, :],
                            prior_log_odds, self.obj_concrete_temp,
                            self._tensors["obj_log_odds"][:, h, w, b, :], self.obj_concrete_temp
                        )
                    else:
                        prob = self._tensors["obj_prob"][:, h, w, b, :]

                        _obj_kl = (
                            prob * (tf_safe_log(prob) - tf_safe_log(p_z)) +
                            (1-prob) * (tf_safe_log(1-prob) - tf_safe_log(1-p_z))
                        )

                    obj_kl.append(_obj_kl)

                    sample = tf.to_float(self._tensors["obj"][:, h, w, b, :] > 0.5)
                    mult = sample * p_z_given_Cz + (1-sample) * (1-p_z_given_Cz)
                    count_distribution = mult * count_distribution
                    normalizer = tf.reduce_sum(count_distribution, axis=1, keepdims=True)
                    normalizer = tf.maximum(normalizer, 1e-6)
                    count_distribution = count_distribution / normalizer

                    count_so_far += sample

                    i += 1

        if "obj" in self.no_gradient:
            obj_kl = tf.stop_gradient(obj_kl)

        self._tensors["obj_kl"] = tf.reshape(tf.concat(obj_kl, axis=1), (self.batch_size, self.H, self.W, self.B, 1))

        # --- interpret program ---

        self._tensors["n_objects"] = tf.fill((self.batch_size,), self.HWB)
        self._tensors["pred_n_objects"] = tf.reduce_sum(self._tensors['obj'], axis=(1, 2, 3, 4))
        self._tensors["pred_n_objects_hard"] = tf.reduce_sum(tf.round(self._tensors['obj']), axis=(1, 2, 3, 4))

        if self.object_encoder is None:
            self.object_encoder = cfg.build_object_encoder(scope="object_encoder")
            if "encoder" in self.fixed_weights:
                self.object_encoder.fix_variables()

        if self.object_decoder is None:
            self.object_decoder = cfg.build_object_decoder(scope="object_decoder")
            if "decoder" in self.fixed_weights:
                self.object_decoder.fix_variables()

        self._build_program_interpreter()

        # --- specify values to record/summarize ---

        recorded_tensors = {}

        recorded_tensors['batch_size'] = tf.to_float(self.batch_size)
        recorded_tensors['float_is_training'] = self.float_is_training

        recorded_tensors['cell_y'] = tf.reduce_mean(self._tensors["cell_y"])
        recorded_tensors['cell_x'] = tf.reduce_mean(self._tensors["cell_x"])
        recorded_tensors['h'] = tf.reduce_mean(self._tensors["h"])
        recorded_tensors['w'] = tf.reduce_mean(self._tensors["w"])
        recorded_tensors['area'] = tf.reduce_mean(self._tensors["area"])

        recorded_tensors['cell_y_std'] = tf.reduce_mean(self._tensors["cell_y_std"])
        recorded_tensors['cell_x_std'] = tf.reduce_mean(self._tensors["cell_x_std"])
        recorded_tensors['h_std'] = tf.reduce_mean(self._tensors["h_std"])
        recorded_tensors['w_std'] = tf.reduce_mean(self._tensors["w_std"])

        obj = self._tensors["obj"]
        pred_n_objects = self._tensors["pred_n_objects"]

        recorded_tensors['n_objects'] = tf.reduce_mean(pred_n_objects)
        recorded_tensors['on_cell_y_avg'] = tf.reduce_mean(
            tf.reduce_sum(self._tensors["cell_y"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects)
        recorded_tensors['on_cell_x_avg'] = tf.reduce_mean(
            tf.reduce_sum(self._tensors["cell_x"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects)
        recorded_tensors['on_h_avg'] = tf.reduce_mean(
            tf.reduce_sum(self._tensors["h"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects)
        recorded_tensors['on_w_avg'] = tf.reduce_mean(
            tf.reduce_sum(self._tensors["w"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects)
        recorded_tensors['on_area_avg'] = tf.reduce_mean(
            tf.reduce_sum(self._tensors["area"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects)
        recorded_tensors['obj'] = tf.reduce_mean(obj)

        recorded_tensors['latent_area'] = tf.reduce_mean(self._tensors["latent_area"])
        recorded_tensors['latent_hw'] = tf.reduce_mean(self._tensors["latent_hw"])

        recorded_tensors['attr'] = tf.reduce_mean(self._tensors["attr"])

        # --- losses ---

        losses = dict()

        if self.train_reconstruction:
            loss_key = 'xent' if self.xent_loss else 'squared'

            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = loss_builders[loss_key](output, inp)
            losses['reconstruction'] = (
                self.reconstruction_weight *
                tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            losses['obj_kl'] = self.kl_weight * tf_mean_sum(self._tensors["obj_kl"])

            obj = self._tensors["obj"]

            losses['cell_y_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["cell_y_kl"])
            losses['cell_x_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["cell_x_kl"])
            losses['h_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["h_kl"])
            losses['w_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["w_kl"])
            losses['attr_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["attr_kl"])

        # --- other evaluation metrics ---

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["pred_n_objects_hard"]) - self._tensors["n_annotations"]))
            recorded_tensors["count_1norm"] = tf.reduce_mean(count_1norm)
            recorded_tensors["count_error"] = tf.reduce_mean(tf.to_float(count_1norm > 0.5))

        return {
            "tensors": self._tensors,
            "recorded_tensors": recorded_tensors,
            "losses": losses
        }


class YoloAir_RenderHook(object):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        fetched = self._fetch(updater)

        self._plot_reconstruction(updater, fetched)
        self._plot_patches(updater, fetched, 4)

    def _fetch(self, updater):
        feed_dict = updater.data_manager.do_val()

        network = updater.network

        to_fetch = dict()

        to_fetch["obj"] = network._tensors["obj"]
        to_fetch["images"] = network._tensors["inp"]
        to_fetch["output"] = network._tensors["output"]
        to_fetch["objects"] = network._tensors["objects"]
        to_fetch["n_objects"] = network._tensors["n_objects"]
        to_fetch["normalized_box"] = network._tensors["normalized_box"]
        to_fetch["input_glimpses"] = network._tensors["input_glimpses"]

        if "n_annotations" in network._tensors:
            to_fetch["annotations"] = network._tensors["annotations"]
            to_fetch["n_annotations"] = network._tensors["n_annotations"]

        if 'prediction' in network._tensors:
            to_fetch["prediction"] = network._tensors["prediction"]
            to_fetch["targets"] = network._tensors["targets"]

        if "actions" in network._tensors:
            to_fetch["actions"] = network._tensors["actions"]

        to_fetch = {k: v[:self.N] for k, v in to_fetch.items()}

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)

        return fetched

    def _plot_reconstruction(self, updater, fetched):
        images = fetched['images']
        output = fetched['output']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        _, image_height, image_width, _ = images.shape

        obj = fetched['obj'].reshape(self.N, -1)

        box = (
            fetched['normalized_box'] *
            [image_height, image_width, image_height, image_width]
        )
        box = box.reshape(self.N, -1, 4)

        n_annotations = fetched.get("n_annotations", [0] * self.N)
        annotations = fetched.get("annotations", None)

        actions = fetched.get("actions", None)

        sqrt_N = int(np.ceil(np.sqrt(self.N)))

        on_colour = np.array(to_rgb("xkcd:azure"))
        off_colour = np.array(to_rgb("xkcd:red"))
        cutoff = 0.5

        fig, axes = plt.subplots(2*sqrt_N, 2*sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(2*sqrt_N, 2*sqrt_N)
        for n, (pred, gt) in enumerate(zip(output, images)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax1 = axes[2*i, 2*j]
            ax1.imshow(gt, vmin=0.0, vmax=1.0)

            title = ""
            if prediction is not None:
                title += "target={}, prediction={}".format(np.argmax(targets[n]), np.argmax(prediction[n]))
            if actions is not None:
                title += ", actions={}".format(actions[n, 0])
            ax1.set_title(title)

            ax2 = axes[2*i, 2*j+1]
            ax2.imshow(pred, vmin=0.0, vmax=1.0)

            ax3 = axes[2*i+1, 2*j]
            ax3.imshow(pred, vmin=0.0, vmax=1.0)

            ax4 = axes[2*i+1, 2*j+1]
            ax4.imshow(pred, vmin=0.0, vmax=1.0)

            # Plot proposed bounding boxes
            for o, (top, left, height, width) in zip(obj[n], box[n]):
                colour = o * on_colour + (1-o) * off_colour

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor=colour, facecolor='none')
                ax4.add_patch(rect)

                if o > cutoff:
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=1, edgecolor=colour, facecolor='none')
                    ax3.add_patch(rect)

            # Plot true bounding boxes
            for k in range(n_annotations[n]):
                _, top, bottom, left, right = annotations[n][k]

                height = bottom - top
                width = right - left

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax1.add_patch(rect)

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax3.add_patch(rect)

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor="xkcd:yellow", facecolor='none')
                ax4.add_patch(rect)

            for ax in axes.flatten():
                ax.set_axis_off()

        if prediction is None:
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
        else:
            plt.subplots_adjust(left=0, right=1, top=.9, bottom=0, wspace=0.1, hspace=0.2)

        local_step = np.inf if cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
        path = updater.exp_dir.path_for(
            'plots',
            'sampled_reconstruction',
            'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))
        fig.savefig(path)
        plt.close(fig)

        shutil.copyfile(
            path,
            os.path.join(os.path.dirname(path), 'latest_stage{:0>4}.pdf'.format(updater.stage_idx)))

    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        H, W, B = updater.network.H, updater.network.W, updater.network.B

        input_glimpses = fetched.get('input_glimpses', None)
        objects = fetched['objects']
        obj = fetched['obj']

        on_colour = np.array(to_rgb("xkcd:azure"))
        off_colour = np.array(to_rgb("xkcd:red"))

        for idx in range(N):
            fig, axes = plt.subplots(3*H, W*B, figsize=(20, 20))
            axes = np.array(axes).reshape(3*H, W*B)

            for h in range(H):
                for w in range(W):
                    for b in range(B):
                        _obj = obj[idx, h, w, b, 0]

                        ax = axes[3*h, w * B + b]
                        ax.imshow(objects[idx, h, w, b, :, :, :3], vmin=0.0, vmax=1.0)

                        colour = _obj * on_colour + (1-_obj) * off_colour
                        obj_rect = patches.Rectangle(
                            (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=colour)
                        ax.add_patch(obj_rect)

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[3*h+1, w * B + b]
                        ax.imshow(objects[idx, h, w, b, :, :, 3], cmap="gray", vmin=0.0, vmax=1.0)

                        ax.set_title("obj={}, b={}".format(_obj, b))

                        ax = axes[3*h+2, w * B + b]
                        ax.set_title("input glimpse")

                        ax.imshow(input_glimpses[idx, h, w, b, :, :, :], vmin=0.0, vmax=1.0)

            for ax in axes.flatten():
                ax.set_axis_off()

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)

            local_step = np.inf if cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
            path = updater.exp_dir.path_for(
                'plots',
                'sampled_patches', str(idx),
                'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))

            fig.savefig(path)
            plt.close(fig)

            shutil.copyfile(
                path,
                os.path.join(os.path.dirname(path), 'latest_stage{:0>4}.pdf'.format(updater.stage_idx)))
