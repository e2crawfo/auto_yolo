import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
import sonnet as snt
from matplotlib.colors import to_rgb
import matplotlib.patches as patches
import shutil
import os

from dps import cfg
from dps.utils import Param
from dps.utils.tf import ScopedFunction, RNNCell, build_scheduled_value

from auto_yolo.tf_ops import render_sprites
from auto_yolo.models import yolo_air
from auto_yolo.models.core import loss_builders, AP


class BboxCell(RNNCell):
    def __init__(self, components, batch_indices_for_boxes, image_height, image_width):
        self.components = components
        self.batch_indices_for_boxes = batch_indices_for_boxes
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, t, state, scope=None):
        batch_idx = self.batch_indices_for_boxes[t[0, 0]-1]
        nonzero_indices = tf.where(tf.equal(self.components[batch_idx, :, :], t[0, 0]))

        maxs = tf.reduce_max(nonzero_indices, axis=0)
        mins = tf.reduce_min(nonzero_indices, axis=0)

        yt = mins[0] / self.image_height
        xt = mins[1] / self.image_width

        ys = (maxs[0] - mins[0]) / self.image_height
        xs = (maxs[1] - mins[1]) / self.image_width

        return tf.to_float(tf.stack([yt, xt, ys, xs])[None, :]), state

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 4

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, 1), dtype=dtype)


class YoloBaseline_Network(ScopedFunction):
    fixed_weights = Param("")
    no_gradient = Param("")
    attr_prior_mean = Param(0.0)
    attr_prior_std = Param(1.0)
    train_reconstruction = Param(True)
    train_kl = Param(True)
    kl_weight = Param(1.0)
    reconstruction_weight = Param(1.0)

    xent_loss = Param(True)
    A = Param(100, help="Dimension of attribute vector.")
    object_shape = Param()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.attr_prior_mean = build_scheduled_value(
            self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(
            self.attr_prior_std, "attr_prior_std")

        self.reconstruction_weight = build_scheduled_value(
            self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")

        self.eval_funcs = dict(
            AP_at_point_1=AP(0.1),
            AP_at_point_25=AP(0.25),
            AP_at_point_5=AP(0.5),
            AP=AP())

        self.object_encoder = None
        self.object_decoder = None

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        super(YoloBaseline_Network, self).__init__(scope=scope)

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

    def _build_program_generator(self):
        assert len(self.inp.shape) == 4
        mask = tf.reduce_sum(tf.abs(self.inp - self.background), axis=3) >= 1e-3
        components = tf.contrib.image.connected_components(mask)

        total_n_objects = tf.to_int32(tf.reduce_max(components))
        indices = tf.range(1, total_n_objects+1)

        maxs = tf.reduce_max(components, axis=(1, 2))

        # So that we don't pick up zeros.
        for_mins = tf.where(mask, components, (total_n_objects + 1) * tf.ones_like(components))
        mins = tf.reduce_min(for_mins, axis=(1, 2))

        n_objects = tf.to_int32(tf.maximum((maxs - mins) + 1, 0))

        under = indices[None, :] <= maxs[:, None]
        over = indices[None, :] >= mins[:, None]

        both = tf.to_int32(tf.logical_and(under, over))
        batch_indices_for_objects = tf.argmax(both, axis=0)

        assert_valid_batch_indices = tf.Assert(
            tf.reduce_all(tf.equal(tf.reduce_sum(both, axis=0), 1)), [both], name="assert_valid_batch_indices")

        with tf.control_dependencies([assert_valid_batch_indices]):
            batch_indices_for_objects = tf.identity(batch_indices_for_objects)

        cell = BboxCell(components, batch_indices_for_objects, self.image_height, self.image_width)

        object_bboxes, _ = dynamic_rnn(
            cell, indices[:, None, None], initial_state=cell.zero_state(1, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=True)

        # Get rid of dummy batch dim created for dynamic_rnn
        object_bboxes = object_bboxes[:, 0, :]

        obj = tf.sequence_mask(n_objects)
        routing = tf.reshape(tf.to_int32(obj), (-1,))
        routing = tf.cumsum(routing, exclusive=True)
        routing = tf.reshape(routing, tf.shape(obj))
        obj = tf.to_float(obj[:, :, None])

        self.program = dict(obj=obj)
        self._tensors["program"] = self.program

        self._tensors["normalized_box"] = tf.gather(object_bboxes, routing, axis=0)
        self._tensors["obj"] = obj
        self._tensors["n_objects"] = n_objects
        self._tensors["max_objects"] = tf.reduce_max(n_objects)

    def _build_program_interpreter(self):
        # --- Get object attributes using object encoder ---

        max_objects = self._tensors["max_objects"]

        yt, xt, ys, xs = tf.split(self._tensors["normalized_box"], 4, axis=-1)

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)

        _boxes = tf.concat([xs, 2*(xt + xs/2) - 1, ys, 2*(yt + ys/2) - 1], axis=-1)
        _boxes = tf.reshape(_boxes, (self.batch_size * max_objects, 4))
        grid_coords = warper(_boxes)
        grid_coords = tf.reshape(grid_coords, (self.batch_size, max_objects, *self.object_shape, 2,))
        input_glimpses = tf.contrib.resampler.resampler(self.inp, grid_coords)

        self._tensors["input_glimpses"] = tf.reshape(
            input_glimpses, (self.batch_size, max_objects, *self.object_shape, self.image_depth))

        object_encoder_in = tf.reshape(
            input_glimpses, (self.batch_size * max_objects, *self.object_shape, self.image_depth))

        attr = self.object_encoder(object_encoder_in, (1, 1, 2*self.A), self.is_training)

        attr = tf.reshape(attr, (self.batch_size, max_objects, 2*self.A))

        attr_mean, attr_log_std = tf.split(attr, [self.A, self.A], axis=-1)
        attr_std = tf.exp(attr_log_std)

        attr, attr_kl = yolo_air.normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

        if "attr" in self.no_gradient:
            attr = tf.stop_gradient(attr)
            attr_kl = tf.stop_gradient(attr_kl)

        self._tensors["attr"] = attr
        self._tensors["attr_kl"] = attr_kl

        object_decoder_in = tf.reshape(attr, (self.batch_size * max_objects, 1, 1, self.A))

        # --- Compute sprites from attr using object decoder ---

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth,), self.is_training)

        objects = tf.nn.sigmoid(tf.clip_by_value(object_logits, -10., 10.))

        self._tensors["objects"] = tf.reshape(
            objects, (self.batch_size, max_objects, *self.object_shape, self.image_depth,))

        objects = tf.reshape(objects, (self.batch_size, max_objects, *self.object_shape, self.image_depth,))
        alpha = self._tensors["obj"][:, :, :, None, None] * tf.ones_like(objects[:, :, :, :, :1])
        objects = tf.concat([objects, alpha], axis=-1)

        # -- Reconstruct image ---

        scales = tf.concat([ys, xs], axis=-1)
        scales = tf.reshape(scales, (self.batch_size, max_objects, 2))

        offsets = tf.concat([yt, xt], axis=-1)
        offsets = tf.reshape(offsets, (self.batch_size, max_objects, 2))

        output = render_sprites.render_sprites(
            objects,
            self._tensors["n_objects"],
            scales,
            offsets,
            self._tensors["background"]
        )

        output = tf.clip_by_value(output, 1e-6, 1-1e-6)
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

        self._build_program_generator()

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

        recorded_tensors = dict(
            batch_size=tf.to_float(self.batch_size),
            float_is_training=self.float_is_training,
            n_objects=tf.reduce_mean(self._tensors["n_objects"]),
            attr=tf.reduce_mean(self._tensors["attr"])
        )

        # --- losses ---

        losses = dict()

        if self.train_reconstruction:
            loss_key = 'xent' if self.xent_loss else 'squared'

            obj = self._tensors["obj"]

            output = obj[:, :, :, None, None] * self._tensors["objects"]
            inp = obj[:, :, :, None, None] * self._tensors["input_glimpses"]
            self._tensors['per_pixel_reconstruction_loss'] = loss_builders[loss_key](output, inp)
            losses['reconstruction'] = (
                self.reconstruction_weight *
                tf.reduce_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            losses['attr_kl'] = self.kl_weight * tf.reduce_sum(obj * self._tensors["attr_kl"])

        # --- other evaluation metrics

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(tf.abs(tf.to_int32(self._tensors["n_objects"]) - self._tensors["n_annotations"]))
            recorded_tensors["count_1norm"] = tf.reduce_mean(count_1norm)
            recorded_tensors["count_error"] = tf.reduce_mean(tf.to_float(count_1norm > 0.5))

        return dict(
            tensors=self._tensors,
            recorded_tensors=recorded_tensors,
            losses=losses
        )


class YoloBaseline_MathNetwork(YoloBaseline_Network):
    math_weight = Param()
    largest_digit = Param()

    math_input_network = None
    math_network = None

    @property
    def n_classes(self):
        return self.largest_digit + 1

    def build_math_representation(self, math_attr):
        # Use raw_obj so that there is no discrepancy between validation and train
        return self._tensors["raw_obj"] * math_attr

    def build_graph(self, *args, **kwargs):
        with tf.variable_scope("reconstruction", reuse=self.initialized):
            result = super(YoloBaseline_MathNetwork, self).build_graph(*args, **kwargs)

        if self.math_input_network is None:
            self.math_input_network = cfg.build_math_input(scope="math_input_network")

            if "math" in self.fixed_weights:
                self.math_input_network.fix_variables()

        attr = tf.reshape(self.program['attr'], (self.batch_size * self.HWB, self.A))
        math_attr = self.math_input_network(attr, self.A, self.is_training)
        math_attr = tf.reshape(math_attr, (self.batch_size, self.H, self.W, self.B, self.A))
        self._tensors["math_attr"] = math_attr

        _inp = self.build_math_representation(math_attr)

        if self.math_network is None:
            self.math_network = cfg.build_math_network(scope="math_network")

            if "math" in self.fixed_weights:
                self.math_network.fix_variables()

        logits = self.math_network(_inp, self.n_classes, self.is_training)

        if self.math_weight is not None:
            result["recorded_tensors"]["raw_loss_math"] = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._tensors["targets"],
                    logits=logits
                )
            )

            result["losses"]["math"] = self.math_weight * result["recorded_tensors"]["raw_loss_math"]

        self._tensors["prediction"] = tf.nn.softmax(logits)

        result["recorded_tensors"]["math_accuracy"] = tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.argmax(logits, axis=1),
                    tf.argmax(self._tensors['targets'], axis=1)
                )
            )
        )

        result["recorded_tensors"]["math_1norm"] = tf.reduce_mean(
            tf.to_float(
                tf.abs(tf.argmax(logits, axis=1) - tf.argmax(self._tensors['targets'], axis=1))
            )
        )

        result["recorded_tensors"]["math_correct_prob"] = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits) * self._tensors['targets'], axis=1)
        )

        return result


class YoloBaseline_RenderHook(yolo_air.YoloAir_RenderHook):
    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        input_glimpses = fetched.get('input_glimpses', None)
        objects = fetched['objects']
        obj = fetched['obj']
        n_objects = obj.sum(axis=(1, 2)).astype('i')

        on_colour = np.array(to_rgb("xkcd:azure"))
        off_colour = np.array(to_rgb("xkcd:red"))

        for idx in range(N):
            no = n_objects[idx]
            fig, axes = plt.subplots(2, no, figsize=(20, 20))
            axes = np.array(axes).reshape(2, no)

            for i in range(no):
                _obj = obj[idx, i, 0]
                ax = axes[0, i]
                ax.imshow(objects[idx, i, :, :, :], vmin=0.0, vmax=1.0)

                colour = _obj * on_colour + (1-_obj) * off_colour
                obj_rect = patches.Rectangle(
                    (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=colour)
                ax.add_patch(obj_rect)

                ax = axes[1, i]
                ax.set_title("input glimpse")

                ax.imshow(input_glimpses[idx, i, :, :, :], vmin=0.0, vmax=1.0)

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
