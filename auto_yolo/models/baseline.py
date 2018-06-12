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

from auto_yolo.models import yolo_air, core


class BboxCell(RNNCell):
    def __init__(self, components, batch_indices_for_boxes, image_height, image_width):
        self.components = components
        self.batch_indices_for_boxes = batch_indices_for_boxes
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, t, state, scope=None):
        batch_idx = self.batch_indices_for_boxes[t]
        nonzero_indices = tf.where(tf.equal(self.components[batch_idx, :, :], t))

        maxs = tf.reduce_max(nonzero_indices, axis=0)
        mins = tf.reduce_min(nonzero_indices, axis=0)

        yt = mins[1] / self.image_height
        xt = mins[0] / self.image_width

        ys = (maxs[1] - mins[1]) / self.image_height
        xs = (maxs[0] - mins[0]) / self.image_width

        return tf.stack([yt, xt, ys, xs])

    @property
    def state_size(self):
        return ()

    @property
    def output_size(self):
        return 4

    def zero_state(self, batch_size, dtype):
        return ()


class YoloBaseline_Network(ScopedFunction):
    fixed_weights = Param("")
    no_gradient = Param("")
    attr_prior_mean = Param(0.0)
    attr_prior_std = Param(1.0)
    train_reconstruction = Param(True)
    train_kl = Param(True)
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

        self.eval_funcs = dict(mAP=core.mAP)

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
            input_glimpses, (self.batch_size, self.max_objects, *self.object_shape, self.image_depth))

        object_encoder_in = tf.reshape(
            input_glimpses, (self.batch_size * max_objects, *self.object_shape, self.image_depth))

        attr = self.object_encoder(object_encoder_in, (1, 1, 2*self.A), self.is_training)

        attr = tf.reshape(attr, (self.batch_size, self.H, self.W, self.B, 2*self.A))

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
        assert len(inp.shape) == 4
        mask = tf.reduce_sum(tf.abs(inp - background), axis=3) >= 1e-6
        components = tf.contrib.image.connected_components(mask)

        total_n_objects = tf.to_int32(tf.reduce_max(components))
        indices = tf.range(total_n_objects)

        maxs = tf.reduce_max(components, axis=(1, 2))
        for_mins = tf.where(mask, components, total_n_objects + 1)  # So that we don't pick up zeros.
        mins = tf.reduce_min(for_mins, axis=(1, 2))
        n_objects = tf.maximum((maxs - mins) + 1, 0)

        under = indices[None, :] <= maxs
        over = indices[None, :] >= mins

        both = tf.to_int32(tf.logical_and(under, over))
        batch_indices_for_objects = tf.argmax(both, axis=0)

        assert_valid_batch_indices = tf.Assert(
            tf.reduce_all(tf.equal(tf.reduce_sum(both, axis=0), 1)), [both], name="assert_valid_batch_indices")

        with tf.control_dependencies([assert_valid_batch_indices]):
            batch_indices_for_objects = tf.identity(batch_indices_for_objects)

        cell = BboxCell(components, batch_indices_for_objects)

        object_bboxes, _ = dynamic_rnn(
            cell, indices, initial_state=(),
            parallel_iterations=1, swap_memory=False, time_major=True)

        obj = tf.sequence_mask(n_objects)
        routing = tf.reshape(obj, (-1,))
        routing = tf.cumsum(obj, exclusive=True)
        routing = tf.reshape(routing, tf.shape(obj))

        self._tensors["normalized_box"] = tf.gather(object_bboxes, routing, axis=0)
        self._tensors["obj"] = obj[:, :, None]
        self._tensors["n_objects"] = n_objects
        self._tensors["max_objects"] = tf.reduce_max(n_objects)

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
            n_objects=tf.reduce_mean(n_objects),
            attr=tf.reduce_mean(self._tensors["attr"])
        )

        # --- losses ---

        losses = dict()

        if self.train_reconstruction:
            loss_key = 'xent' if self.xent_loss else 'squared'

            output = obj[:, :, None, None] * self._tensors["objects"]
            inp = obj[:, :, None, None] * self._tensors["input_glimpses"]
            self._tensors['per_pixel_reconstruction_loss'] = core.loss_builders[loss_key](output, inp)
            losses['reconstruction'] = self.reconstruction_weight * tf.reduce_sum(self._tensors['per_pixel_reconstruction_loss'])

        if self.train_kl:
            losses['attr_kl'] = self.kl_weight * tf.reduce_sum(self._tensors["attr_kl"])

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


class YoloBaseline_RenderHook(yolo_air.YoloAir_RenderHook):
    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        input_glimpses = fetched.get('input_glimpses', None)
        objects = fetched['objects']
        obj = fetched['obj']
        n_objects = obj.sum(axis=(1, 2))

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
