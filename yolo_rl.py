import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import k_means
import collections
import sonnet as snt

from dps import cfg
from dps.datasets import EmnistObjectDetectionDataset
from dps.updater import Updater
from dps.utils import Config, Param, Parameterized, prime_factors
from dps.utils.tf import (
    FullyConvolutional, build_gradient_train_op, trainable_variables,
    tf_mean_sum, build_scheduled_value, MLP, masked_mean)
from dps.env.advanced.yolo import mAP
from dps.tf_ops import render_sprites
from dps.updater import DataManager


class Env(object):
    def __init__(self):
        train = EmnistObjectDetectionDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = EmnistObjectDetectionDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class Backbone(FullyConvolutional):
    pixels_per_cell = Param()
    kernel_size = Param()
    n_channels = Param()
    n_final_layers = Param(2)

    def __init__(self, **kwargs):
        sh = sorted(prime_factors(self.pixels_per_cell[0]))
        sw = sorted(prime_factors(self.pixels_per_cell[1]))
        assert max(sh) <= 4
        assert max(sw) <= 4

        if len(sh) < len(sw):
            sh = sh + [1] * (len(sw) - len(sh))
        elif len(sw) < len(sh):
            sw = sw + [1] * (len(sh) - len(sw))

        layout = [dict(filters=self.n_channels, kernel_size=4, strides=(_sh, _sw), padding="RIGHT_ONLY")
                  for _sh, _sw in zip(sh, sw)]

        # These layers don't change the shape
        layout += [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME")
            for i in range(self.n_final_layers)]

        super(Backbone, self).__init__(layout, check_output_shape=True, **kwargs)


class InverseBackbone(FullyConvolutional):
    pixels_per_cell = Param()
    kernel_size = Param()
    n_channels = Param()
    n_final_layers = Param(2)

    def __init__(self, **kwargs):
        # These layers don't change the shape
        layout = [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME", transpose=True)
            for i in range(self.n_final_layers)]

        sh = sorted(prime_factors(self.pixels_per_cell[0]))
        sw = sorted(prime_factors(self.pixels_per_cell[1]))
        assert max(sh) <= 4
        assert max(sw) <= 4

        if len(sh) < len(sw):
            sh = sh + [1] * (len(sw) - len(sh))
        elif len(sw) < len(sh):
            sw = sw + [1] * (len(sh) - len(sw))

        layout += [dict(filters=self.n_channels, kernel_size=4, strides=(_sh, _sw), padding="SAME", transpose=True)
                   for _sh, _sw in zip(sh, sw)]

        super(InverseBackbone, self).__init__(layout, check_output_shape=True, **kwargs)


class NewBackbone(FullyConvolutional):
    pixels_per_cell = Param()
    max_object_shape = Param()
    n_channels = Param()
    n_base_layers = Param(3)
    n_final_layers = Param(2)

    kernel_size = Param()

    def __init__(self, **kwargs):
        receptive_field_shape = (
            self.max_object_shape[0] + self.pixels_per_cell[0],
            self.max_object_shape[1] + self.pixels_per_cell[1],
        )
        cumulative_filter_shape = (
            receptive_field_shape[0] + self.n_base_layers - 1,
            receptive_field_shape[1] + self.n_base_layers - 1,
        )

        layout = []

        for i in range(self.n_base_layers):
            fh = cumulative_filter_shape[0] // self.n_base_layers
            if i < cumulative_filter_shape[0] % self.n_base_layers:
                fh += 1

            fw = cumulative_filter_shape[1] // self.n_base_layers
            if i < cumulative_filter_shape[1] % self.n_base_layers:
                fw += 1

            layout.append(
                dict(filters=self.n_channels, kernel_size=(fh, fw), padding="VALID", strides=1))

        layout.append(dict(filters=self.n_channels, kernel_size=1, padding="VALID", strides=self.pixels_per_cell))

        layout += [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME")
            for i in range(self.n_final_layers)]

        super(NewBackbone, self).__init__(layout, check_output_shape=True, **kwargs)

    def _call(self, inp, output_size, is_training):
        mod = int(inp.shape[1]) % self.pixels_per_cell[0]
        bottom_padding = self.pixels_per_cell[0] - mod if mod > 0 else 0

        padding_h = int(np.ceil(self.max_object_shape[0] / 2))

        mod = int(inp.shape[2]) % self.pixels_per_cell[1]
        right_padding = self.pixels_per_cell[1] - mod if mod > 0 else 0

        padding_w = int(np.ceil(self.max_object_shape[1] / 2))

        padding = [[0, 0], [padding_h, bottom_padding + padding_h], [padding_w, right_padding + padding_w], [0, 0]]

        inp = tf.pad(inp, padding)

        return super(NewBackbone, self)._call(inp, output_size, is_training)


class NextStep(FullyConvolutional):
    kernel_size = Param()
    n_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME"),
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME"),
        ]
        super(NextStep, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder(FullyConvolutional):
    n_decoder_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=4, kernel_size=4, strides=1, padding="SAME", transpose=True),
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder28x28(FullyConvolutional):
    n_decoder_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=4, kernel_size=4, strides=2, padding="SAME", transpose=True),
        ]
        super(ObjectDecoder28x28, self).__init__(layout, check_output_shape=True, **kwargs)


def reconstruction_cost(_tensors, updater):
    pp_reconstruction_loss = _tensors['per_pixel_reconstruction_loss']
    batch_size = tf.shape(pp_reconstruction_loss)[0]
    pp_reconstruction_loss = tf.reshape(pp_reconstruction_loss, (batch_size, -1))
    reconstruction_loss = tf.reduce_sum(pp_reconstruction_loss, axis=1)
    return tf.reshape(reconstruction_loss, (batch_size, 1, 1, 1, 1))


def local_reconstruction_cost(_tensors, network):
    centers_h = (tf.range(network.H, dtype=tf.float32) + 0.5) * network.pixels_per_cell[0]
    centers_w = (tf.range(network.W, dtype=tf.float32) + 0.5) * network.pixels_per_cell[1]

    loc_h = tf.range(network.image_height, dtype=tf.float32)[..., None] + 0.5
    loc_w = tf.range(network.image_width, dtype=tf.float32)[..., None] + 0.5

    dist_h = tf.abs(loc_h - centers_h)
    dist_w = tf.abs(loc_w - centers_w)

    all_filtered = []

    loss = _tensors['per_pixel_reconstruction_loss']

    for b in range(network.B):
        max_distance_h = network.pixels_per_cell[0] / 2 + network.max_hw * network.anchor_boxes[b, 0] / 2
        max_distance_w = network.pixels_per_cell[1] / 2 + network.max_hw * network.anchor_boxes[b, 1] / 2

        # Rectangle filtering
        filt_h = tf.to_float(dist_h < max_distance_h)
        filt_w = tf.to_float(dist_w < max_distance_w)

        signal = tf.reduce_sum(loss, axis=3)

        signal = tf.tensordot(signal, filt_w, [[2], [0]])
        signal = tf.tensordot(signal, filt_h, [[1], [0]])
        signal = tf.transpose(signal, (0, 2, 1))

        all_filtered.append(signal)

    return tf.stack(all_filtered, axis=3)[..., None]


def tf_local_filter(signal, neighbourhood_size):
    if neighbourhood_size == 0:
        return signal

    if neighbourhood_size is None:
        return tf.reduce_sum(signal, axis=(1, 2, 3, 4), keepdims=True)

    _, H, W, B, _ = signal.shape
    H, W, B = int(H), int(W), int(B)

    dist_h = tf.abs(tf.range(H, dtype=tf.float32) - tf.range(H, dtype=tf.float32)[..., None])
    dist_w = tf.abs(tf.range(W, dtype=tf.float32) - tf.range(W, dtype=tf.float32)[..., None])

    # Rectangle filtering
    filt_h = tf.to_float(dist_h <= neighbourhood_size)
    filt_w = tf.to_float(dist_w <= neighbourhood_size)

    # Sum over box and channel dimensions
    signal = tf.reduce_sum(signal, axis=(3, 4))

    signal = tf.tensordot(signal, filt_w, [[2], [0]])
    signal = tf.tensordot(signal, filt_h, [[1], [0]])
    signal = tf.transpose(signal, (0, 2, 1))

    return signal[..., None, None]


class NonzeroCost(object):
    def __init__(self, target_nonzero, neighbourhood_size=0):
        self.target_nonzero = target_nonzero
        self.neighbourhood_size = neighbourhood_size

    def __call__(self, _tensors, network):
        # The order of operations is switched intentionally here...the target applies to the overall number
        # of objects, rather than individual object-ness values.
        return tf.abs(tf_local_filter(_tensors['program']['obj'], self.neighbourhood_size) - self.target_nonzero)


class AreaCost(object):
    def __init__(self, target_area, neighbourhood_size):
        self.target_area = target_area
        self.neighbourhood_size = neighbourhood_size

    def __call__(self, _tensors, updater):
        selected_area_cost = _tensors['program']['obj'] * tf.abs(_tensors['latent_area'] - self.target_area)
        return tf_local_filter(selected_area_cost, self.neighbourhood_size)


class HeightWidthCost(object):
    def __init__(self, target_hw, neighbourhood_size):
        self.target_hw = target_hw
        self.neighbourhood_size = neighbourhood_size

    def __call__(self, _tensors, updater):
        selected_hw_cost = _tensors['program']['obj'] * tf.abs(_tensors['latent_hw'] - self.target_hw)
        return tf_local_filter(selected_hw_cost, self.neighbourhood_size)


def yolo_rl_mAP(_tensors, updater):
    network = updater.network

    obj = _tensors['program']['obj']
    top, left, height, width = np.split(_tensors['normalized_box'], 4, axis=-1)
    annotations = _tensors["annotations"]
    n_annotations = _tensors["n_annotations"]

    batch_size = obj.shape[0]

    top = network.image_height * top
    height = network.image_height * height
    bottom = top + height

    left = network.image_width * left
    width = network.image_width * width
    right = left + width

    ground_truth_boxes = []
    predicted_boxes = []

    for idx in range(batch_size):
        _a = [[0, *rest] for (cls, *rest), _ in zip(annotations[idx], range(n_annotations[idx]))]
        ground_truth_boxes.append(_a)

        _predicted_boxes = []

        for i in range(network.H):
            for j in range(network.W):
                for b in range(network.B):
                    o = obj[idx, i, j, b, 0]

                    if o > 0.0:
                        _predicted_boxes.append(
                            [0, o,
                             top[idx, i, j, b, 0],
                             bottom[idx, i, j, b, 0],
                             left[idx, i, j, b, 0],
                             right[idx, i, j, b, 0]])

        predicted_boxes.append(_predicted_boxes)

    return mAP(predicted_boxes, ground_truth_boxes, 1, iou_threshold=[0.5])


yolo_rl_mAP.keys_accessed = "normalized_box program:obj annotations n_annotations"


def build_xent_loss(predictions, targets):
    return -(
        targets * tf.log(predictions) +
        (1. - targets) * tf.log(1. - predictions))


def build_squared_loss(predictions, targets):
    return (predictions - targets)**2


def build_1norm_loss(predictions, targets):
    return tf.abs(predictions - targets)


loss_builders = {
    'xent': build_xent_loss,
    'squared': build_squared_loss,
    '1norm': build_1norm_loss,
}


class YoloRL_Network(Parameterized):
    pixels_per_cell = Param()
    A = Param(help="Dimension of attribute vector.")
    anchor_boxes = Param(help="List of (h, w) pairs.")
    object_shape = Param()

    use_input_attention = Param()
    decoder_logit_scale = Param()

    min_hw = Param()
    max_hw = Param()

    min_yx = Param(0.0)
    max_yx = Param(1.0)

    box_std = Param()
    attr_std = Param()
    z_std = Param()
    obj_exploration = Param()
    obj_default = Param()
    explore_during_val = Param()

    n_backbone_features = Param()
    n_passthrough_features = Param()

    xent_loss = Param()

    reconstruction_weight = Param(1)
    rl_weight = Param()
    z_weight = Param(1.0)
    use_baseline = Param()

    area_weight = Param()
    hw_weight = Param()
    nonzero_weight = Param()

    local_reconstruction_cost = Param()

    area_neighbourhood_size = Param()
    hw_neighbourhood_size = Param()
    nonzero_neighbourhood_size = Param()

    target_area = Param(0.)
    target_hw = Param(0.)
    target_nonzero = Param(0.)

    fixed_values = Param()
    fixed_weights = Param()
    no_gradient = Param("")
    order = Param()

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

        self.anchor_boxes = np.array(self.anchor_boxes)

        self.COST_funcs = {}

        if self.reconstruction_weight is not None:
            self.reconstruction_weight = build_scheduled_value(self.reconstruction_weight, "reconstruction_weight")
            _reconstruction_cost_func = local_reconstruction_cost if self.local_reconstruction_cost else reconstruction_cost
            self.COST_funcs['reconstruction'] = (self.reconstruction_weight, _reconstruction_cost_func, "both")

        if self.nonzero_weight is not None:
            self.nonzero_weight = build_scheduled_value(self.nonzero_weight, "nonzero_weight")
            self.target_nonzero = build_scheduled_value(self.target_nonzero, "target_nonzero")
            _nonzero_cost_func = NonzeroCost(self.target_nonzero, self.nonzero_neighbourhood_size)
            self.COST_funcs['nonzero'] = (self.nonzero_weight, _nonzero_cost_func, "obj")

        if self.area_weight is not None:
            self.area_weight = build_scheduled_value(self.area_weight, "area_weight")
            self.target_area = build_scheduled_value(self.target_area, "target_area")
            _area_cost_func = AreaCost(self.target_area, self.area_neighbourhood_size)
            self.COST_funcs['area'] = (self.area_weight, _area_cost_func, "obj")

        if self.hw_weight is not None:
            self.hw_weight = build_scheduled_value(self.hw_weight, "hw_weight")
            self.target_hw = build_scheduled_value(self.target_hw, "target_hw")
            _hw_cost_func = HeightWidthCost(self.target_hw, self.hw_neighbourhood_size)
            self.COST_funcs['hw'] = (self.hw_weight, _hw_cost_func, "obj")

        if self.rl_weight is not None:
            self.rl_weight = build_scheduled_value(self.rl_weight, "rl_weight")

        if self.z_weight is not None:
            self.z_weight = build_scheduled_value(self.z_weight, "z_weight")

        self.eval_funcs = dict(mAP=yolo_rl_mAP)

        self.object_encoder = None
        self.object_decoder = None

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        if isinstance(self.order, str):
            self.order = self.order.split()

        if self.use_input_attention:
            assert set(self.order) == set("box obj z".split())
        else:
            assert set(self.order) == set("box obj z attr".split())

        self.backbone = None
        self.layer_params = dict(
            box=dict(
                rep_builder=self._build_box,
                fixed="box" in self.fixed_weights,
                output_size=4,
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
            z=dict(
                rep_builder=self._build_z,
                fixed="z" in self.fixed_weights,
                output_size=1,
                sample_size=1,
                network=None,
                sigmoid=True,
            ),
            attr=dict(
                rep_builder=self._build_attr,
                fixed="attr" in self.fixed_weights,
                output_size=self.A,
                sample_size=self.A,
                network=None,
                sigmoid=False,
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

    @property
    def float_do_explore(self):
        return self._tensors["float_do_explore"]

    def trainable_variables(self, for_opt):
        scoped_functions = (
            [self.object_decoder] +
            [self.layer_params[kind]["network"] for kind in self.order]
        )

        if self.object_encoder is not None:
            scoped_functions.append(self.object_encoder)

        scoped_functions.append(self.backbone)

        tvars = []
        for sf in scoped_functions:
            tvars.extend(trainable_variables(sf.scope, for_opt=for_opt))

        if self.sequential_cfg['on'] and "edge" not in self.fixed_weights:
            tvars.append(self.edge_weights)

        return tvars

    def _process_cost(self, cost):
        assert len(cost.shape) == 5
        return tf.stop_gradient(cost) * tf.ones((1, self.H, self.W, self.B, 1))

    def _build_routing(self):
        """ Compute a routing matrix based on sampled program, for use in program interpretation step.

        Returns
        -------
        max_objects: int
            Maximum number of objects in a single batch element.
        n_objects: (batch_size,) ndarray
            Number of objects in each batch element.
        routing: (batch_size, max_objects, 4) ndarray
            Routing array used as input to tf.gather_nd when interpreting program to form an image.

        """
        flat_obj = tf.stop_gradient(tf.reshape(self.program['obj'], (self.batch_size, -1)))
        n_objects = tf.to_int32(tf.reduce_sum(flat_obj, axis=1))
        max_objects = tf.reduce_max(n_objects)
        flat_z = tf.stop_gradient(tf.reshape(self.program['z'], (self.batch_size, -1)))

        # We want all the "on" z's to be at the front of the array, but with lower z
        # nearer to the start of the array. So we give "off" values a very high z number,
        # and then negate the whole array before running top_k.
        _flat_z = -tf.where(flat_obj > 0.5, flat_z, 1000. * tf.ones_like(flat_z))
        _, indices = tf.nn.top_k(_flat_z, k=max_objects, sorted=True)

        indices += tf.range(self.batch_size)[:, None] * tf.shape(flat_obj)[1]
        indices = tf.reshape(indices, (-1,))  # tf.unravel_index requires a vector, can't handle anything higher-d
        routing = tf.unravel_index(indices, (self.batch_size, self.H, self.W, self.B))
        routing = tf.reshape(routing, (4, self.batch_size, max_objects))
        routing = tf.transpose(routing, [1, 2, 0])

        # sanity check

        routed_obj = tf.gather_nd(self.program['obj'], routing)[..., 0]
        mask = tf.sequence_mask(n_objects)
        masked_routed_obj = tf.boolean_mask(routed_obj, mask)

        assert_valid_routing = tf.Assert(
            tf.reduce_all(masked_routed_obj > 0.5), [routing], name="assert_valid_routing")

        with tf.control_dependencies([assert_valid_routing]):
            _routing = tf.identity(routing)

        self._tensors["routing"] = _routing
        self._tensors["max_objects"] = max_objects
        self._tensors["n_objects"] = n_objects

    def _get_scheduled_value(self, name):
        scalar = self._tensors.get(name, None)
        if scalar is None:
            schedule = getattr(self, name)
            scalar = self._tensors[name] = build_scheduled_value(schedule, name)
        return scalar

    def _build_box(self, box_logits, is_training):
        box = tf.nn.sigmoid(tf.clip_by_value(box_logits, -10., 10.))
        cell_y, cell_x, h, w = tf.split(box, 4, axis=-1)

        assert self.max_yx > self.min_yx
        cell_y = float(self.max_yx - self.min_yx) * cell_y + self.min_yx
        cell_x = float(self.max_yx - self.min_yx) * cell_x + self.min_yx

        assert self.max_hw > self.min_hw

        h = float(self.max_hw - self.min_hw) * h + self.min_hw
        w = float(self.max_hw - self.min_hw) * w + self.min_hw

        if "cell_y" in self.fixed_values:
            cell_y = float(self.fixed_values["cell_y"]) * tf.ones_like(cell_y, dtype=tf.float32)
        if "cell_x" in self.fixed_values:
            cell_x = float(self.fixed_values["cell_x"]) * tf.ones_like(cell_x, dtype=tf.float32)
        if "h" in self.fixed_values:
            h = float(self.fixed_values["h"]) * tf.ones_like(h, dtype=tf.float32)
        if "w" in self.fixed_values:
            w = float(self.fixed_values["w"]) * tf.ones_like(w, dtype=tf.float32)

        if "cell_y" in self.no_gradient:
            cell_y = tf.stop_gradient(cell_y)
        if "cell_x" in self.no_gradient:
            cell_x = tf.stop_gradient(cell_x)
        if "h" in self.no_gradient:
            h = tf.stop_gradient(h)
        if "w" in self.no_gradient:
            w = tf.stop_gradient(w)

        box = tf.concat([cell_y, cell_x, h, w], axis=-1)

        box_std = self._get_scheduled_value("box_std")

        box_noise = tf.random_normal(tf.shape(box), name="box_noise")

        noisy_box = box + box_noise * box_std * self.float_do_explore

        return dict(
            cell_y=cell_y,
            cell_x=cell_x,
            h=h,
            w=w,
            samples=box_noise,
            program=noisy_box,
        )

    def _build_obj(self, obj_logits, is_training, **kwargs):
        obj_logits = tf.clip_by_value(obj_logits, -10., 10.)

        obj_params = tf.nn.sigmoid(obj_logits)

        obj_exploration = self.float_do_explore * self._get_scheduled_value("obj_exploration")
        obj_default = self._get_scheduled_value("obj_default")

        obj_params = (1 - obj_exploration) * obj_params + obj_default * obj_exploration

        obj_dist = tf.distributions.Bernoulli(probs=obj_params)

        obj_samples = tf.stop_gradient(obj_dist.sample())
        obj_samples = tf.to_float(obj_samples)

        if "obj" in self.fixed_values:
            obj_samples = float(self.fixed_values["obj"]) * tf.ones_like(obj_samples, dtype=tf.float32)

        obj_log_probs = obj_dist.log_prob(obj_samples)
        obj_log_probs = tf.where(tf.is_nan(obj_log_probs), -100.0 * tf.ones_like(obj_log_probs), obj_log_probs)

        if "obj" in self.no_gradient:
            obj_log_probs = tf.stop_gradient(obj_log_probs)

        obj_entropy = obj_dist.entropy()

        return dict(
            samples=obj_samples,
            entropy=obj_entropy,
            log_probs=obj_log_probs,
            logits=obj_logits,
            program=obj_samples
        )

    def _build_z(self, z_logits, is_training):
        z_logits = tf.clip_by_value(z_logits, -10., 10.)

        z_params = tf.nn.sigmoid(z_logits)
        z_std = self.float_do_explore * self._get_scheduled_value("z_std")

        z_dist = tf.distributions.Normal(loc=z_params, scale=z_std)

        z_samples = tf.stop_gradient(z_dist.sample())

        if "z" in self.fixed_values:
            z_samples = float(self.fixed_values["z"]) * tf.ones_like(z_samples, dtype=tf.float32)

        z_log_probs = z_dist.log_prob(z_samples)
        z_log_probs = tf.where(tf.is_nan(z_log_probs), -100.0 * tf.ones_like(z_log_probs), z_log_probs)

        if "z" in self.no_gradient:
            z_log_probs = tf.stop_gradient(z_log_probs)

        z_entropy = z_dist.entropy()

        return dict(
            samples=z_samples,
            entropy=z_entropy,
            log_probs=z_log_probs,
            logits=z_logits,
            program=z_samples
        )

    def _build_attr(self, attr, is_training):
        attr_std = self._get_scheduled_value("attr_std")

        attr_noise = tf.random_normal(tf.shape(attr), name="attr_noise")

        noisy_attr = attr + attr_noise * attr_std * self.float_do_explore

        return dict(
            samples=attr_noise,
            program=noisy_attr,
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

        total_output_size = sum(self.layer_params[kind]["output_size"] for kind in self.order)

        self.edge_weights = tf.get_variable("edge_weights", shape=(1, total_output_size), dtype=tf.float32)
        sizes = [self.layer_params[kind]['output_size'] for kind in self.order]
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
                    assert program[h, w, b].shape[1] == total_output_size

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

        scales = tf.gather_nd(tf.concat([ys, xs], axis=-1), self._tensors["routing"])
        offsets = tf.gather_nd(tf.concat([yt, xt], axis=-1), self._tensors["routing"])

        # --- Compute sprites from attrs using object decoder ---

        if self.use_input_attention:
            transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
            warper = snt.AffineGridWarper(
                (self.image_height, self.image_width), self.object_shape, transform_constraints)

            _scales = tf.reshape(scales, (self.batch_size * self._tensors["max_objects"], 2))
            _offsets = tf.reshape(offsets, (self.batch_size * self._tensors["max_objects"], 2))

            _ys, _xs = tf.split(_scales, 2, axis=1)
            _yt, _xt = tf.split(_offsets, 2, axis=1)

            _yt += _ys / 2
            _xt += _xs / 2

            _boxes = tf.concat([_xs, 2*_xt-1, _ys, 2*_yt-1], axis=1)
            grid_coords = warper(_boxes)
            grid_coords = tf.reshape(grid_coords, (self.batch_size, self._tensors["max_objects"],) + self.object_shape + (2,))
            input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)

            self._tensors["input_glimpses"] = input_glimpses

            object_encoder_in = tf.reshape(
                input_glimpses,
                (self.batch_size * self._tensors["max_objects"], self.object_shape[0] * self.object_shape[1] * self.image_depth))

            attrs = self.object_encoder(object_encoder_in, self.A, self.is_training)
            object_decoder_in = attrs
            # object_decoder_in = tf.reshape(attrs, (self.batch_size * self._tensors["max_objects"], 1, 1, self.A))
        else:
            attrs = self.program['attr']
            routed_attrs = tf.gather_nd(attrs, self._tensors["routing"])
            object_decoder_in = routed_attrs
            # object_decoder_in = tf.reshape(routed_attrs, (self.batch_size * self._tensors["max_objects"],  self.A))

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape[0] * self.object_shape[1] * (self.image_depth+1), self.is_training)

        objects = tf.nn.sigmoid(
            self.decoder_logit_scale * tf.clip_by_value(object_logits, -10., 10.))
        # objects = tf.reshape(objects, self.batch_size + self.object_shape + (self.image_depth + 1,))

        objects = tf.reshape(
            objects, (self.batch_size, self._tensors["max_objects"]) + self.object_shape + (self.image_depth+1,))

        if "alpha" in self.fixed_values:
            obj_img, obj_alpha = tf.split(objects, [3, 1], axis=-1)
            fixed_obj_alpha = float(self.fixed_values["alpha"]) * tf.ones_like(obj_alpha, dtype=tf.float32)
            objects = tf.concat([obj_img, fixed_obj_alpha], axis=-1)

        if "alpha" in self.no_gradient:
            obj_img, obj_alpha = tf.split(objects, [3, 1], axis=-1)
            obj_alpha = tf.stop_gradient(obj_alpha)
            objects = tf.concat([obj_img, fixed_obj_alpha], axis=-1)

        self._tensors["objects"] = objects

        # --- Compose images ---

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

        self._tensors['area'] = (
            (ys * float(self.image_height)) * (xs * float(self.image_width)))

        self._tensors['output'] = output

    def _process_labels(self, labels):
        self._tensors.update(
            annotations=labels[0],
            n_annotations=labels[1],
            targets=labels[2],
        )

    def build_graph(self, inp, labels, is_training, background):

        # --- initialize containers for storing outputs ---

        self._tensors = dict(
            samples=dict(),
            entropy=dict(),
            log_probs=dict(),
            logits=dict(),
            program=dict(),
        )

        self.info_types = list(self._tensors.keys())

        self.program = self._tensors["program"]
        self.samples = self._tensors["samples"]
        self.log_probs = self._tensors["log_probs"]

        self._tensors.update(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            background=background,
            float_do_explore=tf.to_float(True if self.explore_during_val else is_training),
            batch_size=tf.shape(inp)[0],
        )

        self._process_labels(labels)

        # --- build graph ---

        if self.sequential_cfg['on']:
            self._build_program_generator_sequential()
        else:
            self._build_program_generator()

        self._build_routing()

        if self.use_input_attention and self.object_encoder is None:
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
        recorded_tensors['float_do_explore'] = self.float_do_explore

        recorded_tensors['cell_y'] = tf.reduce_mean(self._tensors["cell_y"])
        recorded_tensors['cell_x'] = tf.reduce_mean(self._tensors["cell_x"])
        recorded_tensors['h'] = tf.reduce_mean(self._tensors["h"])
        recorded_tensors['w'] = tf.reduce_mean(self._tensors["w"])
        recorded_tensors['area'] = tf.reduce_mean(self._tensors["area"])

        obj = self._tensors["program"]["obj"]
        recorded_tensors['on_cell_y'] = masked_mean(self._tensors["cell_y"], obj)
        recorded_tensors['on_cell_x'] = masked_mean(self._tensors["cell_x"], obj)
        recorded_tensors['on_h'] = masked_mean(self._tensors["h"], obj)
        recorded_tensors['on_w'] = masked_mean(self._tensors["w"], obj)
        recorded_tensors['on_area'] = masked_mean(self._tensors["area"], obj)

        recorded_tensors['obj'] = tf.reduce_mean(obj)
        recorded_tensors['obj_entropy'] = tf.reduce_mean(self._tensors["entropy"]["obj"])
        recorded_tensors['obj_log_probs'] = tf.reduce_mean(self._tensors["log_probs"]["obj"])
        recorded_tensors['obj_logits'] = tf.reduce_mean(self._tensors["logits"]["obj"])

        recorded_tensors['z'] = tf.reduce_mean(self._tensors["program"]["z"])
        recorded_tensors['z_entropy'] = tf.reduce_mean(self._tensors["entropy"]["z"])
        recorded_tensors['z_log_probs'] = tf.reduce_mean(self._tensors["log_probs"]["z"])
        recorded_tensors['z_logits'] = tf.reduce_mean(self._tensors["logits"]["z"])

        if 'attr' in self.order:
            recorded_tensors['attr'] = tf.reduce_mean(self._tensors["program"]["attr"])

        recorded_tensors['latent_area'] = tf.reduce_mean(self._tensors["latent_area"])
        recorded_tensors['latent_hw'] = tf.reduce_mean(self._tensors["latent_hw"])

        # --- required for computing the reconstruction COST ---

        loss_key = 'xent' if self.xent_loss else 'squared'

        if self.reconstruction_weight is not None:
            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = loss_builders[loss_key](output, inp)

        # --- compute rl loss ---

        COST = tf.zeros((self.batch_size, self.H, self.W, self.B, 1))
        COST_obj = tf.zeros((self.batch_size, self.H, self.W, self.B, 1))
        COST_z = tf.zeros((self.batch_size, self.H, self.W, self.B, 1))

        for name, (weight, func, kind) in self.COST_funcs.items():
            if weight is None:
                continue

            cost = self._process_cost(func(self._tensors, self))
            weighted_cost = weight * cost

            if kind == "both":
                COST += weighted_cost
            elif kind == "obj":
                COST_obj += weighted_cost
            elif kind == "z":
                COST_z += weighted_cost
            else:
                raise Exception("NotImplemented")

            recorded_tensors["COST_{}".format(name)] = tf.reduce_mean(cost)
            recorded_tensors["WEIGHTED_COST_{}".format(name)] = tf.reduce_mean(weighted_cost)

        recorded_tensors["TOTAL_COST"] = (
            tf.reduce_mean(COST) +
            tf.reduce_mean(COST_obj) +
            tf.reduce_mean(COST_z)
        )

        if self.use_baseline:
            COST -= tf.reduce_mean(COST, axis=0, keepdims=True)
            COST_obj -= tf.reduce_mean(COST_obj, axis=0, keepdims=True)
            COST_z -= tf.reduce_mean(COST_z, axis=0, keepdims=True)

        rl_loss_map = (COST_obj + COST) * self.log_probs['obj']

        if self.z_weight is not None:
            rl_loss_map += self.z_weight * (COST_z + COST) * self.log_probs['z']

        # --- losses ---

        losses = dict()

        if self.reconstruction_weight is not None:
            recorded_tensors['raw_loss_reconstruction'] = tf_mean_sum(
                self._tensors['per_pixel_reconstruction_loss'])
            losses['reconstruction'] = self.reconstruction_weight * recorded_tensors['raw_loss_reconstruction']

        if self.area_weight is not None:
            recorded_tensors['raw_loss_area'] = tf_mean_sum(
                tf.abs(self._tensors['latent_area'] - self.target_area) * self._tensors['program']['obj'])
            losses['area'] = self.area_weight * recorded_tensors['raw_loss_area']

        if self.hw_weight is not None:
            recorded_tensors['raw_loss_hw'] = tf_mean_sum(
                tf.abs(self._tensors['latent_hw'] - self.target_hw) * self._tensors['program']['obj'])
            losses['hw'] = self.hw_weight * recorded_tensors['raw_loss_hw']

        if self.rl_weight is not None:
            recorded_tensors['raw_loss_rl'] = tf_mean_sum(rl_loss_map)
            losses['rl'] = self.rl_weight * recorded_tensors['raw_loss_rl']

        # --- other evaluation metrics

        recorded_tensors["count_error"] = tf.reduce_mean(
            1 - tf.to_float(
                tf.equal(self._tensors["n_objects"], self._tensors["n_annotations"])
            )
        )

        recorded_tensors["count_1norm"] = tf.reduce_mean(
            tf.to_float(
                tf.abs(self._tensors["n_objects"] - self._tensors["n_annotations"])
            )
        )

        return {
            "tensors": self._tensors,
            "recorded_tensors": recorded_tensors,
            "losses": losses
        }


class Evaluator(object):
    def __init__(self, functions, tensors, updater):
        self.functions = functions
        self.updater = updater

        if not functions:
            self.fetches = {}
            return

        self.placeholders = {name: tf.placeholder(tf.float32, ()) for name in functions.keys()}
        self.summary_op = tf.summary.merge([tf.summary.scalar(k, v) for k, v in self.placeholders.items()])

        fetch_keys = set()
        for f in functions.values():
            for key in f.keys_accessed.split():
                fetch_keys.add(key)

        fetches = {}
        for key in list(fetch_keys):
            dst = fetches
            src = tensors
            subkeys = key.split(":")

            for i, _key in enumerate(subkeys):
                if i == len(subkeys)-1:
                    dst[_key] = src[_key]
                else:
                    if _key not in dst:
                        dst[_key] = dict()
                    dst = dst[_key]
                    src = src[_key]

        self.fetches = fetches

    def eval(self, fetched):
        if not self.functions:
            return {}, b''

        record = {}
        feed_dict = {}
        for name, func in self.functions.items():
            record[name] = np.mean(func(fetched, self.updater))
            feed_dict[self.placeholders[name]] = record[name]

        sess = tf.get_default_session()
        summary = sess.run(self.summary_op, feed_dict=feed_dict)

        return record, summary


class YoloRL_Updater(Updater):
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()
    pretrain = Param()
    pretrain_cfg = Param()

    eval_modes = "val".split()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.network = cfg.build_network(env)
        self.datasets = env.datasets

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

    @property
    def completion(self):
        return self.datasets['train'].completion

    def trainable_variables(self, for_opt):
        return self.network.trainable_variables(for_opt)

    def _update(self, batch_size, collect_summaries):
        feed_dict = self.data_manager.do_train()

        sess = tf.get_default_session()
        record = {}
        summary = b''
        if collect_summaries:
            _, record, summary = sess.run(
                [self.train_op, self.recorded_tensors, self.summary_op], feed_dict=feed_dict)
        else:
            sess.run([self.train_op], feed_dict=feed_dict)

        return dict(train=(record, summary))

    def _evaluate(self, batch_size, mode):
        assert mode in self.eval_modes

        feed_dict = self.data_manager.do_val()

        record = collections.defaultdict(float)
        summary = b''
        n_points = 0

        sess = tf.get_default_session()

        while True:
            try:
                _record, summary, eval_fetched = sess.run(
                    [self.recorded_tensors, self.summary_op, self.evaluator.fetches], feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

            eval_record, eval_summary = self.evaluator.eval(eval_fetched)
            _record.update(eval_record)
            summary = summary + eval_summary

            batch_size = _record['batch_size']

            for k, v in _record.items():
                record[k] += batch_size * v

            n_points += batch_size

        for k, v in record.items():
            record[k] /= n_points

        return record, summary

    def _build_graph(self):
        self.data_manager = DataManager(self.datasets['train'],
                                        self.datasets['val'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        inp, *labels = self.data_manager.iterator.get_next()

        if cfg.background_cfg.mode == "static":
            with cfg.background_cfg.static_cfg:
                from kmodes.kmodes import KModes
                print("Clustering...")
                print(cfg.background_cfg.static_cfg)

                cluster_data = self.datasets["train"].X
                image_shape = cluster_data.shape[1:]
                indices = np.random.choice(
                    cluster_data.shape[0], replace=False, size=cfg.n_clustering_examples)
                cluster_data = cluster_data[indices, ...]
                cluster_data = cluster_data.reshape(cluster_data.shape[0], -1)

                if cfg.use_k_modes:
                    km = KModes(n_clusters=cfg.n_clusters, init='Huang', n_init=1, verbose=1)
                    km.fit(cluster_data)
                    centroids = km.cluster_centroids_ / 255.
                else:
                    cluster_data = cluster_data / 255.
                    result = k_means(cluster_data, cfg.n_clusters)
                    centroids = result[0]

                centroids = np.maximum(centroids, 1e-6)
                centroids = np.minimum(centroids, 1-1e-6)
                centroids = centroids.reshape(cfg.n_clusters, *image_shape)
        elif cfg.background_cfg.mode == "mode":
            self.background = self.inp_mode[:, None, None, :] * tf.ones_like(inp)
        else:
            self.background = tf.zeros_like(inp)

        if self.pretrain:
            self.network.set_pretraining_params(self.pretrain_cfg)

        network_outputs = self.network(
            (inp, labels, self.background), 0, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        # For running functions, during evaluation, that are not implemented in tensorflow
        self.evaluator = Evaluator(self.network.eval_funcs, network_tensors, self)

        recorded_tensors = {}

        output = network_tensors["output"]
        recorded_tensors.update({
            "loss_" + name: tf_mean_sum(builder(output, inp))
            for name, builder in loss_builders.items()
        })

        recorded_tensors['loss'] = 0
        for name, tensor in network_losses.items():
            recorded_tensors['loss'] += tensor
            recorded_tensors['loss_' + name] = tensor
        self.loss = recorded_tensors['loss']

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        self.recorded_tensors = recorded_tensors

        _summary = [tf.summary.scalar(name, t) for name, t in recorded_tensors.items()]

        # --- train op ---

        tvars = self.trainable_variables(for_opt=True)

        self.train_op, train_summary = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.summary_op = tf.summary.merge(_summary + train_summary)


class YoloRL_RenderHook(object):
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
        to_fetch["annotations"] = network._tensors["annotations"]
        to_fetch["n_annotations"] = network._tensors["n_annotations"]
        to_fetch["output"] = network._tensors["output"]
        to_fetch["objects"] = network._tensors["objects"]
        to_fetch["routing"] = network._tensors["routing"]
        to_fetch["n_objects"] = network._tensors["n_objects"]
        to_fetch["normalized_box"] = network._tensors["normalized_box"]

        if network.use_input_attention:
            to_fetch["input_glimpses"] = network._tensors["input_glimpses"]

        to_fetch = {k: v[:self.N] for k, v in to_fetch.items()}

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)

        return fetched

    def _plot_reconstruction(self, updater, fetched):
        images = fetched['images']
        output = fetched['output']

        _, image_height, image_width, _ = images.shape
        H, W, B = updater.network.H, updater.network.W, updater.network.B

        obj = fetched['obj'].reshape(self.N, H*W*B)

        box = (
            fetched['normalized_box'] *
            [image_height, image_width, image_height, image_width]
        )
        box = box.reshape(self.N, H*W*B, 4)

        annotations = fetched["annotations"]
        n_annotations = fetched["n_annotations"]

        sqrt_N = int(np.ceil(np.sqrt(self.N)))

        fig, axes = plt.subplots(2*sqrt_N, 2*sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(2*sqrt_N, 2*sqrt_N)
        for n, (pred, gt) in enumerate(zip(output, images)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax1 = axes[2*i, 2*j]
            ax1.imshow(gt, vmin=0.0, vmax=1.0)

            ax2 = axes[2*i, 2*j+1]
            ax2.imshow(pred, vmin=0.0, vmax=1.0)

            ax3 = axes[2*i+1, 2*j]
            ax3.imshow(pred, vmin=0.0, vmax=1.0)

            ax4 = axes[2*i+1, 2*j+1]
            ax4.imshow(pred, vmin=0.0, vmax=1.0)

            # Plot proposed bounding boxes
            for o, (top, left, height, width) in zip(obj[n], box[n]):
                color = "xkcd:azure" if o > 1e-6 else "xkcd:red"

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1, edgecolor=color, facecolor='none')
                ax4.add_patch(rect)

                if o > 1e-6:
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=1, edgecolor=color, facecolor='none')
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

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)

        local_step = np.inf if cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
        path = updater.exp_dir.path_for(
            'plots',
            'sampled_reconstruction',
            'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))
        fig.savefig(path)

        plt.close(fig)

    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        H, W, B = updater.network.H, updater.network.W, updater.network.B

        input_glimpses = fetched.get('input_glimpses', None)
        objects = fetched['objects']
        obj = fetched['obj']
        n_objects = fetched['n_objects']
        routing = fetched['routing']
        z = fetched['z']

        for idx in range(N):
            fig, axes = plt.subplots(3*H, W*B, figsize=(20, 20))
            axes = np.array(axes).reshape(3*H, W*B)

            for h in range(H):
                for w in range(W):
                    for b in range(B):
                        _obj = obj[idx, h, w, b, 0]
                        _z = z[idx, h, w, b, 0]

                        ax = axes[3*h, w * B + b]
                        ax.set_aspect('equal')

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[3*h+1, w * B + b]
                        ax.set_aspect('equal')

                        ax.set_title("obj={}, z={}, b={}".format(_obj, _z, b))

                        ax = axes[3*h+2, w * B + b]
                        ax.set_aspect('equal')
                        ax.set_title("input glimpse")

            for i in range(n_objects[idx]):
                _, h, w, b = routing[idx, i]

                ax = axes[3*h, w * B + b]

                ax.imshow(objects[idx, i, :, :, :3], vmin=0.0, vmax=1.0)

                ax = axes[3*h+1, w * B + b]
                ax.imshow(objects[idx, i, :, :, 3], cmap="gray", vmin=0.0, vmax=1.0)

                if input_glimpses is not None:
                    ax = axes[3*h+2, w * B + b]
                    ax.imshow(input_glimpses[idx, i, :, :, :], vmin=0.0, vmax=1.0)

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)

            local_step = np.inf if cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
            path = updater.exp_dir.path_for(
                'plots',
                'sampled_patches', str(idx),
                'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))

            fig.savefig(path)
            plt.close(fig)


xkcd_colors = 'viridian,cerulean,vermillion,lavender,celadon,fuchsia,saffron,cinnamon,greyish,vivid blue'.split(',')


# env config

config = Config(
    log_name="yolo_rl",
    build_env=Env,
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


config.update(
    get_updater=YoloRL_Updater,
    build_network=YoloRL_Network,

    lr_schedule=1e-4,
    batch_size=32,

    optimizer_spec="adam",
    use_gpu=True,
    gpu_allow_growth=True,
    preserve_env=True,
    stopping_criteria="TOTAL_COST,min",
    eval_mode="val",
    threshold=-np.inf,
    max_grad_norm=1.0,
    max_experiments=None,

    eval_step=100,
    display_step=1000,
    render_step=5000,

    max_steps=1e7,
    patience=10000,

    render_hook=YoloRL_RenderHook(),

    # network params

    build_backbone=NewBackbone,
    build_next_step=NextStep,
    build_object_encoder=lambda scope: MLP([100, 100], scope=scope),
    build_object_decoder=lambda scope: MLP([100, 100], scope=scope),
    # build_object_decoder=ObjectDecoder,

    use_input_attention=False,
    decoder_logit_scale=10.0,

    pixels_per_cell=(12, 12),
    max_object_shape=(28, 28),

    anchor_boxes=[[14, 14]],

    kernel_size=(1, 1),

    n_channels=128,
    n_decoder_channels=128,
    A=100,

    n_backbone_features=100,
    n_passthrough_features=0,

    min_hw=0.3,
    max_hw=3.0,

    box_std=0.0,
    attr_std=0.0,
    z_std=0.1,
    obj_exploration=0.05,
    obj_default=0.5,
    explore_during_val=False,

    rl_weight=1.0,

    area_weight=1.0,
    hw_weight=None,
    nonzero_weight=1.0,

    use_baseline=True,

    local_reconstruction_cost=False,
    area_neighbourhood_size=None,
    hw_neighbourhood_size=None,
    nonzero_neighbourhood_size=None,

    fixed_values=dict(),
    fixed_weights="",
    order="box obj z attr",

    sequential_cfg=dict(
        on=True,
        lookback_shape=(2, 2, 2),
        build_next_step=lambda scope: MLP([100, 100], scope=scope),
    ),

    curriculum=[
        dict(rl_weight=0.0, fixed_values=dict(obj=1), max_steps=10000, patience=10000),
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
        dict(obj_exploration=0.03),
        dict(obj_exploration=0.02),
        dict(obj_exploration=0.01),
        dict(do_train=False, n_train=16, min_chars=1, postprocessing="", preserve_env=False),
    ],
)

single_digit_config = config.copy(
    log_name="yolo_rl_single_digit",

    min_chars=1,
    max_chars=1,
    image_shape=(24, 24),
    pixels_per_cell=(12, 12),
    area_weight=0.1,

    postprocessing="",

    render_step=500,

    curriculum=[
        dict(obj_exploration=0.2),
        dict(obj_exploration=0.1),
        dict(obj_exploration=0.05),
        dict(obj_exploration=0.025),
        dict(obj_exploration=0.0125),
        dict(obj_exploration=0.0),
    ]
)
