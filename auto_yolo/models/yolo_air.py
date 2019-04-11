import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps import cfg
from dps.utils import Param
from dps.utils.tf import tf_mean_sum, RenderHook, GridConvNet

from auto_yolo.models.core import AP, xent_loss, VariationalAutoencoder
from auto_yolo.models.object_layer import ObjectLayer


class YoloAir_Network(VariationalAutoencoder):
    n_backbone_features = Param()
    anchor_boxes = Param()

    backbone = None
    object_layer = None

    _eval_funcs = None

    def __init__(self, env, updater, scope=None, **kwargs):
        super(YoloAir_Network, self).__init__(env, updater, scope=scope, **kwargs)
        self.B = len(self.anchor_boxes)

    @property
    def eval_funcs(self):
        if "annotations" in self._tensors:
            if self._eval_funcs is None:
                ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
                eval_funcs["AP"] = AP(ap_iou_values)
                self._eval_funcs = eval_funcs
            return self._eval_funcs
        else:
            return {}

    def build_representation(self):
        # --- build graph ---

        if self.backbone is None:
            self.backbone = cfg.build_backbone(scope="backbone")
            assert isinstance(self.backbone, GridConvNet)
            if "backbone" in self.fixed_weights:
                self.backbone.fix_variables()

        inp = self._tensors["inp"]
        backbone_output, n_grid_cells, grid_cell_size = self.backbone(
            inp, self.B*self.n_backbone_features, self.is_training)

        self.H, self.W = [int(i) for i in n_grid_cells]
        self.HWB = self.H * self.W * self.B
        self.pixels_per_cell = tuple(int(i) for i in grid_cell_size)

        backbone_output = tf.reshape(
            backbone_output,
            (-1, self.H, self.W, self.B, self.n_backbone_features))

        if self.object_layer is None:
            self.object_layer = ObjectLayer(self.pixels_per_cell, scope="objects")

        object_rep_tensors = self.object_layer(
            self.inp, backbone_output, self._tensors["background"], self.is_training)
        self._tensors.update(object_rep_tensors)

        # --- specify values to record ---

        obj = self._tensors["obj"]
        pred_n_objects = self._tensors["pred_n_objects"]

        self.record_tensors(
            batch_size=self.batch_size,
            float_is_training=self.float_is_training,

            cell_y=self._tensors["cell_y"],
            cell_x=self._tensors["cell_x"],
            h=self._tensors["h"],
            w=self._tensors["w"],
            z=self._tensors["z"],
            area=self._tensors["area"],

            cell_y_std=self._tensors["cell_y_std"],
            cell_x_std=self._tensors["cell_x_std"],
            h_std=self._tensors["h_std"],
            w_std=self._tensors["w_std"],
            z_std=self._tensors["z_std"],

            n_objects=pred_n_objects,
            obj=obj,
            on_cell_y_avg=tf.reduce_sum(self._tensors["cell_y"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_cell_x_avg=tf.reduce_sum(self._tensors["cell_x"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_h_avg=tf.reduce_sum(self._tensors["h"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_w_avg=tf.reduce_sum(self._tensors["w"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_z_avg=tf.reduce_sum(self._tensors["z"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_area_avg=tf.reduce_sum(self._tensors["area"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,

            latent_area=self._tensors["latent_area"],
            latent_hw=self._tensors["latent_hw"],

            attr=self._tensors["attr"],
        )

        # --- losses ---

        if self.train_reconstruction:
            output = self._tensors['output']
            inp = self._tensors['inp']
            self._tensors['per_pixel_reconstruction_loss'] = xent_loss(pred=output, label=inp)
            self.losses['reconstruction'] = (
                self.reconstruction_weight * tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])
            )

        if self.train_kl:
            self.losses.update(
                obj_kl=self.kl_weight * tf_mean_sum(self._tensors["obj_kl"]),
                cell_y_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["cell_y_kl"]),
                cell_x_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["cell_x_kl"]),
                h_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["h_kl"]),
                w_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["w_kl"]),
                z_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["z_kl"]),
                attr_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["attr_kl"]),
            )

        # --- other evaluation metrics ---

        if "n_annotations" in self._tensors:
            count_1norm = tf.to_float(
                tf.abs(tf.to_int32(self._tensors["pred_n_objects_hard"]) - self._tensors["n_valid_annotations"]))

            self.record_tensors(
                count_1norm=count_1norm,
                count_error=count_1norm > 0.5,
            )


class YoloAir_RenderHook(RenderHook):
    fetches = "obj raw_obj z inp output objects n_objects normalized_box input_glimpses"

    def __call__(self, updater):
        network = updater.network
        if "n_annotations" in network._tensors:
            self.fetches += " annotations n_annotations"

        if 'prediction' in network._tensors:
            self.fetches += " prediction targets"

        if "actions" in network._tensors:
            self.fetches += " actions"

        fetched = self._fetch(updater)

        try:
            self._plot_reconstruction(updater, fetched)
            self._plot_patches(updater, fetched, 4)
        except Exception:
            pass

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        _, image_height, image_width, _ = inp.shape

        obj = fetched['obj'].reshape(self.N, -1)

        box = (
            fetched['normalized_box']
            * [image_height, image_width, image_height, image_width]
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
        for n, (pred, gt) in enumerate(zip(output, inp)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax1 = axes[2*i, 2*j]
            self.imshow(ax1, gt)

            title = ""
            if prediction is not None:
                title += "target={}, prediction={}".format(np.argmax(targets[n]), np.argmax(prediction[n]))
            if actions is not None:
                title += ", actions={}".format(actions[n, 0])
            ax1.set_title(title)

            ax2 = axes[2*i, 2*j+1]
            self.imshow(ax2, pred)

            ax3 = axes[2*i+1, 2*j]
            self.imshow(ax3, pred)

            ax4 = axes[2*i+1, 2*j+1]
            self.imshow(ax4, pred)

            # Plot proposed bounding boxes
            for o, (top, left, height, width) in zip(obj[n], box[n]):
                colour = o * on_colour + (1-o) * off_colour

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=2, edgecolor=colour, facecolor='none')
                ax4.add_patch(rect)

                if o > cutoff:
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=2, edgecolor=colour, facecolor='none')
                    ax3.add_patch(rect)

            # Plot true bounding boxes
            for k in range(n_annotations[n]):
                valid, _, top, bottom, left, right = annotations[n][k]

                if not valid:
                    continue

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

        self.savefig("sampled_reconstruction", fig, updater)

    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        H, W, B = updater.network.H, updater.network.W, updater.network.B

        input_glimpses = fetched.get('input_glimpses', None)
        objects = fetched['objects']
        obj = fetched['obj']
        raw_obj = fetched['raw_obj']
        z = fetched['z']

        on_colour = np.array(to_rgb("xkcd:azure"))
        off_colour = np.array(to_rgb("xkcd:red"))

        for idx in range(N):
            fig, axes = plt.subplots(3*H, W*B, figsize=(20, 20))
            axes = np.array(axes).reshape(3*H, W*B)

            for h in range(H):
                for w in range(W):
                    for b in range(B):
                        _obj = obj[idx, h, w, b, 0]
                        _raw_obj = raw_obj[idx, h, w, b, 0]
                        _z = z[idx, h, w, b, 0]

                        ax = axes[3*h, w * B + b]
                        self.imshow(ax, objects[idx, h, w, b, :, :, :3])

                        colour = _obj * on_colour + (1-_obj) * off_colour
                        obj_rect = patches.Rectangle(
                            (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=colour)
                        ax.add_patch(obj_rect)

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[3*h+1, w * B + b]
                        self.imshow(ax, objects[idx, h, w, b, :, :, 3], cmap="gray")

                        ax.set_title("obj={}, raw_obj={}, z={}, b={}".format(_obj, _raw_obj, _z, b))

                        ax = axes[3*h+2, w * B + b]
                        ax.set_title("input glimpse")

                        self.imshow(ax, input_glimpses[idx, h, w, b, :, :, :])

            for ax in axes.flatten():
                ax.set_axis_off()

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)

            self.savefig("sampled_patches/" + str(idx), fig, updater)


class YoloAir_ComparisonRenderHook(RenderHook):
    fetches = "obj inp output objects n_objects normalized_box"

    show_zero_boxes = True

    def __call__(self, updater):
        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']

        _, image_height, image_width, _ = inp.shape

        obj = fetched['obj'].reshape(self.N, -1)

        box = (
            fetched['normalized_box']
            * [image_height, image_width, image_height, image_width]
        )
        box = box.reshape(self.N, -1, 4)

        on_colour = np.array(to_rgb("xkcd:azure"))
        off_colour = np.array(to_rgb("xkcd:red"))

        for n, (pred, gt) in enumerate(zip(output, inp)):
            fig = plt.figure(figsize=(5, 5))
            ax = plt.gca()

            self.imshow(ax, gt)
            ax.set_axis_off()

            # Plot proposed bounding boxes
            for o, (top, left, height, width) in zip(obj[n], box[n]):
                if not self.show_zero_boxes and o <= 1e-6:
                    continue

                colour = o * on_colour + (1-o) * off_colour

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=2, edgecolor=colour, facecolor='none')
                ax.add_patch(rect)

            plt.subplots_adjust(left=.01, right=.99, top=.99, bottom=0.01, wspace=0.1, hspace=0.1)
            self.savefig("ground_truth/" + str(n), fig, updater, is_dir=False)


class YoloAir_PaperSetRenderHook(RenderHook):
    fetches = "obj raw_obj z inp output objects n_objects normalized_box input_glimpses"
    do_annotations = True

    def __call__(self, updater):
        self.fetches += " annotations n_annotations"
        fetched = self._fetch(updater)

        try:
            self._plot_reconstruction(updater, fetched)
            self._plot_patches(updater, fetched, self.N)
        except Exception:
            pass

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']

        _, image_height, image_width, _ = inp.shape

        obj = fetched['obj'].reshape(self.N, -1)

        box = (
            fetched['normalized_box']
            * [image_height, image_width, image_height, image_width]
        )
        box = box.reshape(self.N, -1, 4)

        pred_colour = np.array(to_rgb(self.pred_colour))

        if self.do_annotations:
            n_annotations = fetched.get("n_annotations", [0] * self.N)
            annotations = fetched.get("annotations", None)
            gt_colour = np.array(to_rgb(self.gt_colour))

        cutoff = 0.5

        for n, (pred, gt) in enumerate(zip(output, inp)):
            fig, axes = plt.subplots(1, 3, figsize=(6, 3))
            axes = np.array(axes).reshape(3)

            ax1 = axes[0]
            self.imshow(ax1, gt)

            ax2 = axes[1]
            self.imshow(ax2, pred)

            ax3 = axes[2]
            self.imshow(ax3, pred)

            # Plot proposed bounding boxes
            for o, (top, left, height, width) in zip(obj[n], box[n]):
                if o > cutoff:
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=2, edgecolor=pred_colour, facecolor='none')
                    ax3.add_patch(rect)

            if self.do_annotations:
                # Plot true bounding boxes
                for k in range(n_annotations[n]):
                    valid, _, top, bottom, left, right = annotations[n][k]

                    if not valid:
                        continue

                    height = bottom - top
                    width = right - left

                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=1, edgecolor=gt_colour, facecolor='none')
                    ax3.add_patch(rect)

            for ax in axes.flatten():
                ax.set_axis_off()

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)
            self.savefig("sampled_reconstruction/" + str(n), fig, updater)

    def _plot_patches(self, updater, fetched, N):
        # Create a plot showing what each object is generating
        import matplotlib.pyplot as plt

        H, W, B = updater.network.H, updater.network.W, updater.network.B

        input_glimpses = fetched.get('input_glimpses', None)
        objects = fetched['objects']
        obj = fetched['obj']
        raw_obj = fetched['raw_obj']
        z = fetched['z']

        on_colour = np.array(to_rgb("xkcd:azure"))
        off_colour = np.array(to_rgb("xkcd:red"))

        for idx in range(N):
            fig, axes = plt.subplots(3*H, W*B, figsize=(20, 20))
            axes = np.array(axes).reshape(3*H, W*B)

            for h in range(H):
                for w in range(W):
                    for b in range(B):
                        _obj = obj[idx, h, w, b, 0]
                        _raw_obj = raw_obj[idx, h, w, b, 0]
                        _z = z[idx, h, w, b, 0]

                        ax = axes[3*h, w * B + b]
                        self.imshow(ax, objects[idx, h, w, b, :, :, :3])

                        colour = _obj * on_colour + (1-_obj) * off_colour
                        obj_rect = patches.Rectangle(
                            (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=colour)
                        ax.add_patch(obj_rect)

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[3*h+1, w * B + b]
                        self.imshow(ax, objects[idx, h, w, b, :, :, 3], cmap="gray")

                        ax.set_title("obj={}, raw_obj={}, z={}, b={}".format(_obj, _raw_obj, _z, b))

                        ax = axes[3*h+2, w * B + b]
                        ax.set_title("input glimpse")

                        self.imshow(ax, input_glimpses[idx, h, w, b, :, :, :])

            for ax in axes.flatten():
                ax.set_axis_off()

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)

            self.savefig("sampled_patches/" + str(idx), fig, updater)
