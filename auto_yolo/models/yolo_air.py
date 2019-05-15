import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

from dps.utils import Param
from dps.utils.tf import tf_mean_sum, RenderHook, GridConvNet

from auto_yolo.models.core import AP, xent_loss, VariationalAutoencoder, coords_to_pixel_space
from auto_yolo.models.object_layer import GridObjectLayer, ObjectRenderer


class YoloAir_Network(VariationalAutoencoder):
    n_backbone_features = Param()
    n_objects_per_cell = Param()

    backbone = None
    object_layer = None
    object_renderer = None

    _eval_funcs = None

    def __init__(self, env, updater, scope=None, **kwargs):
        super(YoloAir_Network, self).__init__(env, updater, scope=scope, **kwargs)
        self.B = self.n_objects_per_cell

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

        self.maybe_build_subnet("backbone")
        assert isinstance(self.backbone, GridConvNet)

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
            self.object_layer = GridObjectLayer(self.pixels_per_cell, scope="objects")

        if self.object_renderer is None:
            self.object_renderer = ObjectRenderer(scope="renderer")

        objects = self.object_layer(self.inp, backbone_output, self.is_training)
        self._tensors.update(objects)

        kl_tensors = self.object_layer.compute_kl(objects)
        self._tensors.update(kl_tensors)

        render_tensors = self.object_renderer(objects, self._tensors["background"], self.is_training)
        self._tensors.update(render_tensors)

        # --- specify values to record ---

        obj = self._tensors["obj"]
        pred_n_objects = self._tensors["pred_n_objects"]

        self.record_tensors(
            batch_size=self.batch_size,
            float_is_training=self.float_is_training,

            cell_y=self._tensors["cell_y"],
            cell_x=self._tensors["cell_x"],
            height=self._tensors["height"],
            width=self._tensors["width"],
            z=self._tensors["z"],

            cell_y_std=self._tensors["cell_y_logit_dist"].scale,
            cell_x_std=self._tensors["cell_x_logit_dist"].scale,
            height_std=self._tensors["height_logit_dist"].scale,
            width_std=self._tensors["width_logit_dist"].scale,
            z_std=self._tensors["z_logit_dist"].scale,

            n_objects=pred_n_objects,
            obj=obj,
            on_cell_y_avg=tf.reduce_sum(self._tensors["cell_y"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_cell_x_avg=tf.reduce_sum(self._tensors["cell_x"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_height_avg=tf.reduce_sum(self._tensors["height"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_width_avg=tf.reduce_sum(self._tensors["width"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,
            on_z_avg=tf.reduce_sum(self._tensors["z"] * obj, axis=(1, 2, 3, 4)) / pred_n_objects,

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
                height_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["height_kl"]),
                width_kl=self.kl_weight * tf_mean_sum(obj * self._tensors["width_kl"]),
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
    fetches = "render_obj obj z inp output appearance n_objects normalized_box glimpse"

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

        anchor_box = updater.network.object_layer.anchor_box
        yt, xt, ys, xs = np.split(fetched['normalized_box'], 4, axis=-1)
        yt, xt, ys, xs = coords_to_pixel_space(
            yt, xt, ys, xs, (image_height, image_width), anchor_box, top_left=True)
        box = np.concatenate([yt, xt, ys, xs], axis=-1)
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
                valid, _, _, top, bottom, left, right = annotations[n][k]

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

        glimpse = fetched.get('glimpse', None)
        appearance = fetched['appearance']
        obj = fetched['obj']
        render_obj = fetched['render_obj']
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
                        _render_obj = render_obj[idx, h, w, b, 0]
                        _z = z[idx, h, w, b, 0]

                        ax = axes[3*h, w * B + b]
                        self.imshow(ax, appearance[idx, h, w, b, :, :, :3])

                        colour = _obj * on_colour + (1-_obj) * off_colour
                        obj_rect = patches.Rectangle(
                            (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=colour)
                        ax.add_patch(obj_rect)

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[3*h+1, w * B + b]
                        self.imshow(ax, appearance[idx, h, w, b, :, :, 3], cmap="gray")

                        ax.set_title("obj={:.2f}, render_obj={:.2f}, z={:.2f}, b={}".format(_obj, _render_obj, _z, b))

                        ax = axes[3*h+2, w * B + b]
                        ax.set_title("input glimpse")

                        self.imshow(ax, glimpse[idx, h, w, b, :, :, :])

            for ax in axes.flatten():
                ax.set_axis_off()

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)

            self.savefig("sampled_patches/" + str(idx), fig, updater)


class YoloAir_ComparisonRenderHook(RenderHook):
    fetches = "obj inp output appearance n_objects normalized_box"

    show_zero_boxes = True

    def __call__(self, updater):
        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']

        _, image_height, image_width, _ = inp.shape

        obj = fetched['obj'].reshape(self.N, -1)

        anchor_box = updater.network.object_layer.anchor_box
        yt, xt, ys, xs = np.split(fetched['normalized_box'], 4, axis=-1)
        yt, xt, ys, xs = coords_to_pixel_space(
            yt, xt, ys, xs, (image_height, image_width), anchor_box, top_left=True)
        box = np.concatenate([yt, xt, ys, xs], axis=-1)
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
    fetches = "obj render_obj z inp output appearance n_objects normalized_box glimpse"
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

        anchor_box = updater.network.object_layer.anchor_box
        yt, xt, ys, xs = np.split(fetched['normalized_box'], 4, axis=-1)
        yt, xt, ys, xs = coords_to_pixel_space(
            yt, xt, ys, xs, (image_height, image_width), anchor_box, top_left=True)
        box = np.concatenate([yt, xt, ys, xs], axis=-1)
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
                for k in range(n_annotations[n]):
                    valid, _, _, top, bottom, left, right = annotations[n][k]

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

        glimpse = fetched.get('glimpse', None)
        appearance = fetched['appearance']
        obj = fetched['obj']
        render_obj = fetched['render_obj']
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
                        _render_obj = render_obj[idx, h, w, b, 0]
                        _z = z[idx, h, w, b, 0]

                        ax = axes[3*h, w * B + b]
                        self.imshow(ax, appearance[idx, h, w, b, :, :, :3])

                        colour = _obj * on_colour + (1-_obj) * off_colour
                        obj_rect = patches.Rectangle(
                            (1, 0), 0.2, 1, clip_on=False, transform=ax.transAxes, facecolor=colour)
                        ax.add_patch(obj_rect)

                        if h == 0 and b == 0:
                            ax.set_title("w={}".format(w))
                        if w == 0 and b == 0:
                            ax.set_ylabel("h={}".format(h))

                        ax = axes[3*h+1, w * B + b]
                        self.imshow(ax, appearance[idx, h, w, b, :, :, 3], cmap="gray")

                        ax.set_title("obj={:.2f}, render_obj={:.2f}, z={:.2f}, b={}".format(_obj, _render_obj, _z, b))

                        ax = axes[3*h+2, w * B + b]
                        ax.set_title("input glimpse")

                        self.imshow(ax, glimpse[idx, h, w, b, :, :, :])

            for ax in axes.flatten():
                ax.set_axis_off()

            plt.subplots_adjust(left=0.02, right=.98, top=.98, bottom=0.02, wspace=0.1, hspace=0.1)

            self.savefig("sampled_patches/" + str(idx), fig, updater)
