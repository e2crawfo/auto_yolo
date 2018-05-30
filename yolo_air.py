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
from dps.utils import Config, Param
from dps.utils.tf import ScopedFunction, trainable_variables, tf_mean_sum, build_scheduled_value, MLP
from dps.tf_ops import render_sprites
from dps.env.advanced import yolo_rl


def normal_kl(mean, std, prior_mean, prior_std):
    var = std**2
    prior_var = prior_std**2

    return 0.5 * (
        tf.log(prior_var) - tf.log(var) -
        1.0 + var / prior_var +
        tf.square(mean - prior_mean) / prior_var
    )


def normal_vae(mean, std, prior_mean, prior_std):
    sample = mean + tf.random_normal(tf.shape(mean)) * std
    kl = normal_kl(mean, std, prior_mean, prior_std)
    return sample, kl


def concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=10e-10):
    u = tf.random_uniform(tf.shape(log_odds), minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)
    return (log_odds + noise) / temperature


def concrete_binary_sample_kl(pre_sigmoid_sample,
                              prior_log_odds, prior_temperature,
                              posterior_log_odds, posterior_temperature,
                              eps=10e-10):
    y = pre_sigmoid_sample

    y_times_prior_temp = y * prior_temperature
    log_prior = tf.log(prior_temperature + eps) - y_times_prior_temp + prior_log_odds - \
        2.0 * tf.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)

    y_times_posterior_temp = y * posterior_temperature
    log_posterior = tf.log(posterior_temperature + eps) - y_times_posterior_temp + posterior_log_odds - \
        2.0 * tf.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)

    return log_posterior - log_prior


def build_xent_loss(predictions, targets):
    return -(
        targets * tf.log(predictions) +
        (1. - targets) * tf.log(1. - predictions))


def build_squared_loss(predictions, targets):
    return (predictions - targets)**2


def build_normal_ll_loss(predictions, targets):
    return ((predictions - targets)**2) / (0.3**2)


def build_1norm_loss(predictions, targets):
    return tf.abs(predictions - targets)


loss_builders = {
    'xent': build_xent_loss,
    'squared': build_squared_loss,
    '1norm': build_1norm_loss,
}


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
    order = Param("box obj")

    use_concrete_kl = Param(True)
    count_prior_log_odds = Param()
    obj_temperature = Param(1.0)

    train_reconstruction = Param(True)
    train_kl = Param(True)

    yx_prior_mean = Param(0.0)
    yx_prior_std = Param(1.0)

    hw_prior_mean = Param(0.0)
    hw_prior_std = Param(1.0)

    attr_prior_mean = Param(0.0)
    attr_prior_std = Param(1.0)

    obj_logit_scale = Param(2.0)
    alpha_logit_scale = Param(0.1)
    alpha_logit_bias = Param(5.0)

    sequential_cfg = Param(dict(
        on=False,
        lookback_shape=(2, 2, 2),
    ))

    def __init__(self, env, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.H = int(np.ceil(self.image_height / self.pixels_per_cell[0]))
        self.W = int(np.ceil(self.image_width / self.pixels_per_cell[1]))
        self.B = len(self.anchor_boxes)
        self.HWB = self.H * self.W * self.B

        self.count_prior_log_odds = build_scheduled_value(self.count_prior_log_odds, "count_prior_log_odds")
        self.obj_temperature = build_scheduled_value(self.obj_temperature, "obj_temperature")

        self.yx_prior_mean = build_scheduled_value(self.yx_prior_mean, "yx_prior_mean")
        self.yx_prior_std = build_scheduled_value(self.yx_prior_std, "yx_prior_std")

        self.hw_prior_mean = build_scheduled_value(self.hw_prior_mean, "hw_prior_mean")
        self.hw_prior_std = build_scheduled_value(self.hw_prior_std, "hw_prior_std")

        self.attr_prior_mean = build_scheduled_value(self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(self.attr_prior_std, "attr_prior_std")

        self.anchor_boxes = np.array(self.anchor_boxes)

        self.eval_funcs = dict(mAP=yolo_rl.yolo_rl_mAP)

        self.object_encoder = None
        self.object_decoder = None

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        if isinstance(self.order, str):
            self.order = self.order.split()

        assert set(self.order) == set("box obj".split())

        self.backbone = None
        self.layer_params = dict(
            box=dict(
                rep_builder=self._build_box,
                fixed="box" in self.fixed_weights,
                output_size=8,
                sample_size=4,
                network=None,
                sigmoid=True,
            ),
            obj=dict(
                rep_builder=self._build_obj,
                fixed="obj" in self.fixed_weights,
                output_size=1,
                sample_size=1,
                network=None,
                sigmoid=True,
            ),
        )

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

    def trainable_variables(self, for_opt):
        scoped_functions = (
            [self.object_encoder, self.object_decoder, self.backbone] +
            [self.layer_params[kind]["network"] for kind in self.order]
        )

        tvars = []
        for sf in scoped_functions:
            tvars.extend(trainable_variables(sf.scope, for_opt=for_opt))

        if self.sequential_cfg['on'] and "edge" not in self.fixed_weights:
            tvars.append(self.edge_weights)

        return tvars

    def _get_scheduled_value(self, name):
        scalar = self._tensors.get(name, None)
        if scalar is None:
            schedule = getattr(self, name)
            scalar = self._tensors[name] = build_scheduled_value(schedule, name)
        return scalar

    def _build_box(self, box_params, is_training):
        mean, log_std = tf.split(box_params, 2, axis=-1)
        std = tf.exp(log_std)

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
            cell_y=cell_y,
            cell_x=cell_x,
            h=h,
            w=w,

            cell_y_kl=cy_kl,
            cell_x_kl=cx_kl,
            h_kl=h_kl,
            w_kl=w_kl,

            kl=box_kl,
            program=box,
        )

    def _build_obj(self, obj_logits, is_training, **kwargs):
        obj_log_odds = tf.clip_by_value(obj_logits, -10., 10.)

        obj_pre_sigmoid = concrete_binary_pre_sigmoid_sample(
            obj_log_odds, self.obj_temperature
        )
        raw_obj = tf.nn.sigmoid(obj_pre_sigmoid)
        obj = (
            self.float_is_training * raw_obj +
            (1 - self.float_is_training) * tf.round(raw_obj)
        )

        if "obj" in self.no_gradient:
            obj = tf.stop_gradient(obj)

        return dict(
            program=obj,
            raw_obj=raw_obj,
            obj_pre_sigmoid=obj_pre_sigmoid,
            obj_log_odds=obj_log_odds,
            obj_prob=tf.nn.sigmoid(obj_log_odds),
        )

    def _build_program_generator(self):
        H, W, B = self.H, self.W, self.B
        program, features = None, None

        if self.backbone is None:
            self.backbone = cfg.build_backbone(scope="backbone")
            self.backbone.layout[-1]['filters'] = B * self.n_backbone_features

            if "backbone" in self.fixed_weights:
                self.backbone.fix_variables()

        inp = self._tensors["inp"]
        backbone_output = self.backbone(inp, (H, W, B * self.n_backbone_features), self.is_training)

        for i, kind in enumerate(self.order):
            params = self.layer_params[kind]
            rep_builder = params["rep_builder"]
            output_size = params["output_size"]
            network = params["network"]
            fixed = params["fixed"]

            first = i == 0
            final = i == len(self.order) - 1

            n_features = 0 if final else self.n_passthrough_features

            if network is None:
                network = cfg.build_next_step(scope="{}_network".format(kind))
                network.layout[-1]['filters'] = B * output_size + n_features

                if fixed:
                    network.fix_variables()
                self.layer_params[kind]["network"] = network

            if first:
                layer_inp = backbone_output
            else:
                _program = tf.reshape(program, (-1, H, W, B * int(program.shape[-1])))
                layer_inp = tf.concat([backbone_output, features, _program], axis=3)

            network_output = network(layer_inp, (H, W, B * output_size + n_features), self.is_training)

            rep_input, features = tf.split(network_output, (B * output_size, n_features), axis=3)

            rep_input = tf.reshape(rep_input, (-1, H, W, B, output_size))

            built = rep_builder(rep_input, self.is_training)

            assert 'program' in built
            for key, value in built.items():
                if key in self.info_types:
                    self._tensors[key][kind] = value
                else:
                    assert key not in self._tensors, "Overwriting with key `{}`".format(key)
                    self._tensors[key] = value

            if first:
                program = self.program[kind]
            else:
                program = tf.concat([program, self.program[kind]], axis=4)

    def _get_sequential_input(self, program, h, w, b, edge_element):
        inp = []
        for i in range(self.sequential_cfg.lookback_shape[0]):
            for j in range(self.sequential_cfg.lookback_shape[1]):
                for k in range(self.sequential_cfg.lookback_shape[2]):

                    if i == j == k == 0:
                        continue

                    _i = h - i
                    _j = w - j
                    _k = b - k

                    if _i < 0 or _j < 0 or _k < 0:
                        inp.append(edge_element)
                    else:
                        inp.append(program[_i, _j, _k])

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

        total_sample_size = sum(self.layer_params[kind]["sample_size"] for kind in self.order)

        self.edge_weights = tf.get_variable("edge_weights", shape=(1, total_sample_size), dtype=tf.float32)
        sizes = [self.layer_params[kind]['sample_size'] for kind in self.order]
        edge_weights = tf.split(self.edge_weights, sizes, axis=1)
        _edge_weights = []
        for ew, kind in zip(edge_weights, self.order):
            if self.layer_params[kind]['sigmoid']:
                ew = tf.nn.sigmoid(ew)
            _edge_weights.append(ew)
        edge_element = tf.concat(_edge_weights, axis=1)
        edge_element = tf.tile(edge_element, (self.batch_size, 1))

        # --- initialize containers for storing built program ---

        _tensors = collections.defaultdict(self._make_empty)
        _tensors.update({
            info_type: collections.defaultdict(self._make_empty)
            for info_type in self.info_types})

        program = np.empty((H, W, B), dtype=np.object)

        # --- build the program ---

        for h in range(self.H):
            for w in range(self.W):
                for b in range(self.B):
                    partial_program, features = None, None
                    _backbone_output = backbone_output[:, h, w, b, :]
                    context = self._get_sequential_input(program, h, w, b, edge_element)

                    for i, kind in enumerate(self.order):
                        params = self.layer_params[kind]
                        rep_builder = params["rep_builder"]
                        output_size = params["output_size"]
                        network = params["network"]
                        fixed = params["fixed"]

                        first = i == 0
                        final = i == len(self.order) - 1

                        if network is None:
                            network = self.sequential_cfg.build_next_step(scope="{}_sequential_network".format(kind))
                            if fixed:
                                network.fix_variables()
                            params["network"] = network

                        if first:
                            layer_inp = tf.concat([_backbone_output, context], axis=1)
                        else:
                            layer_inp = tf.concat([_backbone_output, context, features, partial_program], axis=1)

                        n_features = 0 if final else self.n_passthrough_features

                        network_output = network(layer_inp, output_size + n_features, self.is_training)

                        rep_input, features = tf.split(network_output, (output_size, n_features), axis=1)

                        built = rep_builder(rep_input, self.is_training)

                        assert 'program' in built
                        for key, value in built.items():
                            if key in self.info_types:
                                _tensors[key][kind][h, w, b] = value
                            else:
                                _tensors[key][h, w, b] = value

                        if first:
                            partial_program = built['program']
                        else:
                            partial_program = tf.concat([partial_program, built['program']], axis=1)

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

        for key, value in list(_tensors.items()):
            if isinstance(value, dict):
                for kind, v in list(value.items()):
                    self._tensors[key][kind] = form_tensor(v)
            else:
                self._tensors[key] = form_tensor(value)

    def _build_program_interpreter(self):

        # --- Compute sprite locations from box parameters ---

        # All in cell-local co-ordinates, should be invariant to image size.
        boxes = self.program['box']
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

        self._tensors["attr"] = attr
        self.program["attr"] = attr
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

        obj_alpha *= tf.reshape(self.program['obj'], (self.batch_size, self.HWB, 1, 1, 1))
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

        output = tf.clip_by_value(output, 1e-6, 1-1e-6)

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

        # --- initialize containers for storing outputs ---

        self._tensors = dict(
            logits=dict(),
            program=dict(),
            kl=dict(),
        )

        self.info_types = list(self._tensors.keys())

        self.program = self._tensors["program"]

        self._tensors.update(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            background=background,
            batch_size=tf.shape(inp)[0],
        )

        self._process_labels(labels)

        # --- build graph ---

        if self.sequential_cfg['on']:
            self._build_program_generator_sequential()
        else:
            self._build_program_generator()

        # --- compute obj_kl ---

        count_support = tf.range(self.HWB+1, dtype=tf.float32)
        count_prior_prob = tf.nn.sigmoid(self.count_prior_log_odds)
        count_distribution = (1 - count_prior_prob) * (count_prior_prob ** count_support)
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
                            prior_log_odds, self.obj_temperature,
                            self._tensors["obj_log_odds"][:, h, w, b, :], self.obj_temperature
                        )
                    else:
                        prob = self._tensors["obj_prob"][:, h, w, b, :]

                        _obj_kl = (
                            prob * (tf_safe_log(prob) - tf_safe_log(p_z)) +
                            (1-prob) * (tf_safe_log(1-prob) - tf_safe_log(1-p_z))
                        )

                    obj_kl.append(_obj_kl)

                    sample = tf.to_float(self.program["obj"][:, h, w, b, :] > 0.5)
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
        self._tensors["pred_n_objects"] = tf.reduce_sum(self.program['obj'], axis=(1, 2, 3, 4))

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

        obj = self._tensors["program"]["obj"]
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
            losses['reconstruction'] = tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])

        if self.train_kl:
            losses['obj_kl'] = tf_mean_sum(self._tensors["obj_kl"])

            obj = self.program["obj"]

            losses['cell_y_kl'] = tf_mean_sum(obj * self._tensors["cell_y_kl"])
            losses['cell_x_kl'] = tf_mean_sum(obj * self._tensors["cell_x_kl"])
            losses['h_kl'] = tf_mean_sum(obj * self._tensors["h_kl"])
            losses['w_kl'] = tf_mean_sum(obj * self._tensors["w_kl"])
            losses['attr_kl'] = tf_mean_sum(obj * self._tensors["attr_kl"])

        # --- other evaluation metrics

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(tf.abs(tf.to_int32(self._tensors["pred_n_objects"]) - self._tensors["n_annotations"]))
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

        to_fetch = network.program.copy()

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
        H, W, B = updater.network.H, updater.network.W, updater.network.B

        obj = fetched['obj'].reshape(self.N, H*W*B)

        box = (
            fetched['normalized_box'] *
            [image_height, image_width, image_height, image_width]
        )
        box = box.reshape(self.N, H*W*B, 4)

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


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


# env config

env_config = Config(
    log_name="yolo_air",

    build_env=yolo_rl.Env,
    seed=347405995,

    min_chars=12,
    max_chars=12,
    n_patch_examples=0,

    image_shape=(84, 84),
    max_overlap=49,
    patch_shape=(14, 14),

    characters=list(range(10)),
    patch_size_std=0.0,
    colours="white",

    n_distractors_per_image=0,

    backgrounds="",
    backgrounds_sample_every=False,
    background_colours="",

    background_cfg=dict(mode="none"),

    object_shape=(14, 14),

    xent_loss=True,

    postprocessing="random",
    n_samples_per_image=4,
    tile_shape=(42, 42),
    max_attempts=1000000,

    preserve_env=True,

    n_train=25000,
    n_val=1e2,
    n_test=1e2,
)


# model config


# This works quite well if it is trained for long enough.
alg_config = Config(
    get_updater=yolo_rl.YoloRL_Updater,
    build_network=YoloAir_Network,

    lr_schedule=1e-4,
    batch_size=32,

    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    preserve_env=True,
    stopping_criteria="loss,min",
    eval_mode="val",
    threshold=-np.inf,
    max_grad_norm=1.0,
    max_experiments=None,

    eval_step=100,
    display_step=1000,
    render_step=500,

    max_steps=50000,
    patience=100000,

    render_hook=YoloAir_RenderHook(),

    # network params

    build_object_encoder=lambda scope: MLP([512, 256], scope=scope),
    build_object_decoder=lambda scope: MLP([256, 512], scope=scope),
    build_next_step=yolo_rl.NextStep,
    # build_backbone=yolo_rl.NewBackbone,
    # max_object_shape=(28, 28),
    # build_object_decoder=ObjectDecoder,
    build_backbone=yolo_rl.Backbone,

    pixels_per_cell=(12, 12),

    kernel_size=(1, 1),

    n_channels=128,
    n_decoder_channels=128,
    A=50,

    sequential_cfg=dict(
        on=True,
        lookback_shape=(2, 2, 2),
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    hw_prior_mean=np.log(0.1/0.9),
    hw_prior_std=1.0,
    anchor_boxes=[[48, 48]],
    count_prior_log_odds="Exp(start=10000.0, end=0.2, decay_rate=0.1, decay_steps=200, log=True)",
    # count_prior_log_odds="Exp(start=10000.0, end=0.000000001, decay_rate=0.1, decay_steps=200, log=True)",
    use_concrete_kl=False,

    overwrite_plots=False,

    curriculum=[
        dict(),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
    ],
)

config = env_config.copy()
config.update(alg_config)

big_single_config = config.copy(
    image_shape=(40, 40),
    postprocessing="",
    curriculum=[
        dict(),
    ],
    object_shape=(28, 28),
    patch_shape=(28, 28),
    max_overlap=2*196,
    min_chars=1,
    max_chars=1,
    anchor_boxes=[[40, 40]],
    hw_prior_std=1.0,
    kernel_size=(3, 3),
    # hw_prior_mean=10.0,

    # build_backbone=yolo_rl.NewBackbone,
    # max_object_shape=(28, 28),
)

big_double_config = config.copy(
    image_shape=(48, 48),
    postprocessing="",
    curriculum=[
        dict(),
    ],
    object_shape=(28, 28),
    patch_shape=(28, 28),
    max_overlap=2*196,
    min_chars=1,
    max_chars=2,
    anchor_boxes=[[40, 40]],
    kernel_size=(3, 3),
    alpha_logit_scale=0.25,
    obj_logit_scale=2.0,
    count_prior_log_odds="Exp(start=10000.0, end=0.2, decay_rate=0.1, decay_steps=200, log=True)",
    hw_prior_std=2.0,
    # hw_prior_mean=10.0,

    build_backbone=yolo_rl.NewBackbone,
    max_object_shape=(28, 28),
)

big_config = config.copy(
    image_shape=(48, 48),
    postprocessing="",
    curriculum=[
        dict(),
    ],
    object_shape=(28, 28),
    patch_shape=(28, 28),
    max_overlap=2*196,
    min_chars=1,
    max_chars=2,
    anchor_boxes=[[48, 48]],
    # fixed_values=dict(alpha=1),
    hw_prior_std=10.0,

    build_backbone=yolo_rl.NewBackbone,
    max_object_shape=(28, 28),
)

big_colour_config = big_config.copy(
    colours="red green blue",
)

colour_config = config.copy(
    colours="red green blue",
)

single_digit_config = config.copy(
    log_name="yolo_rl_single_digit",

    min_chars=1,
    max_chars=1,
    image_shape=(24, 24),
    pixels_per_cell=(12, 12),

    postprocessing="",

    curriculum=[
        dict()
    ]
)
