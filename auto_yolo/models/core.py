import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import k_means
import collections

from dps import cfg
from dps.updater import Updater as _Updater
from dps.env.advanced.yolo import mAP as _mAP
from dps.datasets import EmnistObjectDetectionDataset
from dps.utils import Param, prime_factors
from dps.utils.tf import (
    FullyConvolutional, build_gradient_train_op, tf_mean_sum)
from dps.updater import DataManager


class Env(object):
    def __init__(self):
        train = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = EmnistObjectDetectionDataset(
            n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

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


def mAP(_tensors, updater):
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

    return _mAP(predicted_boxes, ground_truth_boxes, 1, iou_threshold=[0.5])


mAP.keys_accessed = "normalized_box program:obj annotations n_annotations"


class Updater(_Updater):
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


class RenderHook(object):
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
                    (left, top), width, height, linewidth=1,
                    edgecolor=color, facecolor='none')
                ax4.add_patch(rect)

                if o > 1e-6:
                    rect = patches.Rectangle(
                        (left, top), width, height, linewidth=1,
                        edgecolor=color, facecolor='none')
                    ax3.add_patch(rect)

            # Plot true bounding boxes
            for k in range(n_annotations[n]):
                _, top, bottom, left, right = annotations[n][k]

                height = bottom - top
                width = right - left

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1,
                    edgecolor="xkcd:yellow", facecolor='none')
                ax1.add_patch(rect)

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1,
                    edgecolor="xkcd:yellow", facecolor='none')
                ax3.add_patch(rect)

                rect = patches.Rectangle(
                    (left, top), width, height, linewidth=1,
                    edgecolor="xkcd:yellow", facecolor='none')
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
