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
from dps.utils.tf import RNNCell, tf_mean_sum, tf_shape, tf_cosine_similarity

from auto_yolo.tf_ops import render_sprites
from auto_yolo.models import yolo_air
from auto_yolo.models.core import xent_loss, AP, VariationalAutoencoder, normal_vae


class BboxCell(RNNCell):
    def __init__(self, components, batch_indices_for_boxes, image_height, image_width):
        self.components = components
        self.batch_indices_for_boxes = batch_indices_for_boxes
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, t, state, scope=None):
        """ t is the index of the object in the whole batch, batch_idx is the index of the
            batch element that the object belongs to, which we have pre-computed """
        batch_idx = self.batch_indices_for_boxes[t[0, 0]-1]
        nonzero_indices = tf.where(tf.equal(self.components[batch_idx, :, :], t[0, 0]))

        mins = tf.cast(tf.reduce_min(nonzero_indices, axis=0), tf.float32)
        maxs = tf.cast(tf.reduce_max(nonzero_indices, axis=0), tf.float32) + 1

        cyt = (mins[0] + (maxs[0] - mins[0]) / 2) / self.image_height
        cxt = (mins[1] + (maxs[1] - mins[1]) / 2) / self.image_width

        ys = (maxs[0] - mins[0]) / self.image_height
        xs = (maxs[1] - mins[1]) / self.image_width

        return tf.to_float(tf.stack([cyt, cxt, ys, xs])[None, :]), state

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 4

    def zero_state(self, batch_size, dtype):
        return tf.zeros((batch_size, 1), dtype=dtype)


def _find_connected_componenents_body(mask):
    components = tf.contrib.image.connected_components(mask)

    total_n_objects = tf.to_int32(tf.reduce_max(components))
    indices = tf.range(1, total_n_objects + 1)

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
        tf.reduce_all(tf.equal(tf.reduce_sum(both, axis=0), 1)),
        [both], name="assert_valid_batch_indices")

    with tf.control_dependencies([assert_valid_batch_indices]):
        batch_indices_for_objects = tf.identity(batch_indices_for_objects)

    _, image_height, image_width, *_ = tf_shape(mask)
    cell = BboxCell(components, batch_indices_for_objects, image_height, image_width)

    # For each object, get its bounding box by using `indices` to figure out which element of
    # `components` the object appears in, and then check that element
    object_bboxes, _ = dynamic_rnn(
        cell, indices[:, None, None], initial_state=cell.zero_state(1, tf.float32),
        parallel_iterations=10, swap_memory=False, time_major=True)

    # Couldn't I have just iterated through all object indices and used tf.where on `components` to simultaneously
    # get both the bounding box and the batch index? Yes, but I think I thought that would be expensive
    # (have to look through the entirety of `components` once for each object).

    # Get rid of dummy batch dim created for dynamic_rnn
    object_bboxes = object_bboxes[:, 0, :]

    obj = tf.sequence_mask(n_objects)
    routing = tf.reshape(tf.to_int32(obj), (-1,))
    routing = tf.cumsum(routing, exclusive=True)
    routing = tf.reshape(routing, tf.shape(obj))
    obj = tf.to_float(obj[:, :, None])

    return dict(
        normalized_box=tf.gather(object_bboxes, routing, axis=0),
        obj=obj,
        n_objects=n_objects,
        max_objects=tf.reduce_max(n_objects),
    )


def tf_find_connected_components(inp, bg, threshold, colors=None, cosine_threshold=None):
    assert len(inp.shape) == 4

    if isinstance(colors, str):
        colors = colors.split()

    mask = tf.reduce_sum(tf.abs(inp - bg), axis=3) >= threshold

    if colors is None or cosine_threshold is None:
        output = _find_connected_componenents_body(mask)
        output['color'] = output['obj']
        return output

    objects = []

    for color in colors:
        if isinstance(color, str):
            color = tf.constant(to_rgb(color), tf.float32)

        similarity = tf_cosine_similarity(inp, color)
        color_mask = tf.logical_and(similarity >= cosine_threshold, mask)
        objects.append(
            _find_connected_componenents_body(color_mask)
        )

    output = dict(
        normalized_box=tf.concat([o['normalized_box'] for o in objects], axis=1),
        obj=tf.concat([o['obj'] for o in objects], axis=1),
        n_objects=tf.reduce_sum(tf.stack([o['n_objects'] for o in objects], axis=1), axis=1),
        color=tf.concat([float(i+1) * o['obj'] for i, o in enumerate(objects)], axis=1),
    )
    output['max_objects'] = tf.reduce_max(output['n_objects'])

    return output


class Baseline_Network(VariationalAutoencoder):
    cc_threshold = Param()
    object_shape = Param()

    object_encoder = None
    object_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
        self.eval_funcs["AP"] = AP(ap_iou_values)

        super(Baseline_Network, self).__init__(env, updater, scope=scope, **kwargs)

    def _build_program_generator(self, tensors):
        return tf_find_connected_components(tensors['inp'], tensors['background'], self.cc_threshold)

    def _build_program_interpreter(self, tensors):
        # --- Get object attributes using object encoder ---

        max_objects = tensors["max_objects"]

        yt, xt, ys, xs = tf.split(tensors["normalized_box"], 4, axis=-1)

        transform_constraints = snt.AffineWarpConstraints.no_shear_2d()
        warper = snt.AffineGridWarper(
            (self.image_height, self.image_width), self.object_shape, transform_constraints)

        _boxes = tf.concat([xs, 2*(xt + xs/2) - 1, ys, 2*(yt + ys/2) - 1], axis=-1)
        _boxes = tf.reshape(_boxes, (self.batch_size * max_objects, 4))
        grid_coords = warper(_boxes)
        grid_coords = tf.reshape(grid_coords, (self.batch_size, max_objects, *self.object_shape, 2,))
        glimpse = tf.contrib.resampler.resampler(tensors["inp"], grid_coords)

        object_encoder_in = tf.reshape(
            glimpse, (self.batch_size * max_objects, *self.object_shape, self.image_depth))

        attr = self.object_encoder(object_encoder_in, (1, 1, 2*self.A), self.is_training)
        attr = tf.reshape(attr, (self.batch_size, max_objects, 2*self.A))
        attr_mean, attr_log_std = tf.split(attr, [self.A, self.A], axis=-1)
        attr_std = tf.exp(attr_log_std)

        if not self.noisy:
            attr_std = tf.zeros_like(attr_std)

        attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

        object_decoder_in = tf.reshape(attr, (self.batch_size * max_objects, 1, 1, self.A))

        # --- Compute sprites from attr using object decoder ---

        object_logits = self.object_decoder(
            object_decoder_in, self.object_shape + (self.image_depth,), self.is_training)

        objects = tf.nn.sigmoid(tf.clip_by_value(object_logits, -10., 10.))

        objects = tf.reshape(objects, (self.batch_size, max_objects, *self.object_shape, self.image_depth,))
        alpha = tensors["obj"][:, :, :, None, None] * tf.ones_like(objects[:, :, :, :, :1])
        importance = tf.ones_like(objects[:, :, :, :, :1])
        objects = tf.concat([objects, alpha, importance], axis=-1)

        # -- Reconstruct image ---

        scales = tf.concat([ys, xs], axis=-1)
        scales = tf.reshape(scales, (self.batch_size, max_objects, 2))

        offsets = tf.concat([yt, xt], axis=-1)
        offsets = tf.reshape(offsets, (self.batch_size, max_objects, 2))

        output = render_sprites.render_sprites(
            objects,
            tensors["n_objects"],
            scales,
            offsets,
            tensors["background"]
        )

        return dict(
            output=output,
            glimpse=tf.reshape(
                glimpse, (self.batch_size, max_objects, *self.object_shape, self.image_depth)),
            attr=tf.reshape(attr, (self.batch_size, max_objects, self.A)),
            attr_kl=tf.reshape(attr_kl, (self.batch_size, max_objects, self.A)),
            objects=tf.reshape(
                objects, (self.batch_size, max_objects, *self.object_shape, self.image_depth,))
        )

    def build_representation(self):

        self.maybe_build_subnet("object_encoder")
        self.maybe_build_subnet("object_decoder")

        program_tensors = self._build_program_generator(self._tensors)
        self._tensors.update(program_tensors)

        interpreter_tensors = self._build_program_interpreter(self._tensors)
        self._tensors.update(interpreter_tensors)

        # --- specify values to record ---

        self.record_tensors(
            n_objects=self._tensors["n_objects"],
            attr=self._tensors["attr"]
        )

        # --- losses ---

        if self.train_reconstruction:
            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=output, label=inp)
            self.losses['reconstruction'] = (
                self.reconstruction_weight
                * tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            obj = self._tensors["obj"]
            self.losses['attr_kl'] = self.kl_weight * tf_mean_sum(obj * self._tensors["attr_kl"])

        # --- other evaluation metrics

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["n_objects"]) - self._tensors["n_annotations"]))
            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5
            )


class Baseline_RenderHook(yolo_air.YoloAir_RenderHook):
    fetches = "obj inp output objects n_objects normalized_box glimpse"

    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        glimpse = fetched.get('glimpse', None)
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

                ax.imshow(glimpse[idx, i, :, :, :], vmin=0.0, vmax=1.0)

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
