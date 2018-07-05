import tensorflow as tf
import numpy as np
from sklearn.cluster import k_means
import collections

from dps import cfg
from dps.updater import Updater as _Updater
from dps.env.advanced.yolo import mAP
from dps.datasets import EmnistObjectDetectionDataset
from dps.utils import Param, prime_factors
from dps.utils.tf import (
    FullyConvolutional, build_gradient_train_op, tf_mean_sum)
from dps.updater import DataManager


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

    def __init__(self, check_output_shape=True, **kwargs):
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

        super(Backbone, self).__init__(layout, check_output_shape=check_output_shape, **kwargs)


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
        targets * tf.log(predictions + 1e-9) +
        (1. - targets) * tf.log(1. - predictions + 1e-9))


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
        record = {}
        for name, func in self.functions.items():
            record[name] = np.mean(func(fetched, self.updater))
        return record


class AP(object):
    keys_accessed = "normalized_box obj annotations n_annotations"

    def __init__(self, iou_threshold=None):
        if iou_threshold is not None:
            try:
                iou_threshold = list(iou_threshold)
            except (TypeError, ValueError):
                iou_threshold = [float(iou_threshold)]
        self.iou_threshold = iou_threshold

    def __call__(self, _tensors, updater):
        network = updater.network

        obj = _tensors['obj']
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

        obj = obj.reshape(batch_size, -1)
        top = top.reshape(batch_size, -1)
        bottom = bottom.reshape(batch_size, -1)
        left = left.reshape(batch_size, -1)
        right = right.reshape(batch_size, -1)

        ground_truth_boxes = []
        predicted_boxes = []

        for idx in range(batch_size):
            _a = [
                [0, *rest]
                for (cls, *rest), _
                in zip(annotations[idx], range(n_annotations[idx]))
            ]

            ground_truth_boxes.append(_a)

            _predicted_boxes = []

            for i in range(obj.shape[1]):
                o = obj[idx, i]

                if o > 0.0:
                    _predicted_boxes.append(
                        [0, o,
                         top[idx, i],
                         bottom[idx, i],
                         left[idx, i],
                         right[idx, i]]
                    )

            predicted_boxes.append(_predicted_boxes)

        return mAP(
            predicted_boxes, ground_truth_boxes,
            n_classes=1, iou_threshold=self.iou_threshold)


class Updater(_Updater):
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    eval_modes = "val".split()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.network = cfg.build_network(env, scope="network")
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
        if collect_summaries:
            _, record, train_record = sess.run(
                [self.train_op, self.recorded_tensors, self.train_records], feed_dict=feed_dict)

            summary_feed_dict = {ph: record[k] for k, ph in self.recorded_tensors_ph.items()}
            summary_feed_dict.update({ph: train_record[k] for k, ph in self.train_records_ph.items()})

            summary = sess.run(self.train_summary_op, feed_dict=summary_feed_dict)
        else:
            record = {}
            summary = b''
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
                _record, eval_fetched = sess.run(
                    [self.recorded_tensors, self.evaluator.fetches], feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

            eval_record = self.evaluator.eval(eval_fetched)
            _record.update(eval_record)

            for k, v in _record.items():
                record[k] += batch_size * v

            n_points += batch_size

        for k, v in record.items():
            record[k] /= n_points

        summary_feed_dict = {ph: record[k] for k, ph in self.recorded_tensors_ph.items()}
        summary_feed_dict.update({ph: record[k] for k, ph in self.eval_funcs_ph.items()})

        summary = sess.run(self.val_summary_op, feed_dict=summary_feed_dict)

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

        network_outputs = self.network(
            (inp, labels, self.background), 0, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        self.recorded_tensors = recorded_tensors = dict(global_step=tf.train.get_or_create_global_step())

        # --- loss ---

        recorded_tensors['loss'] = 0
        for name, tensor in network_losses.items():
            recorded_tensors['loss'] += tensor
            recorded_tensors['loss_' + name] = tensor
        self.loss = recorded_tensors['loss']

        # --- train op ---

        tvars = self.trainable_variables(for_opt=True)

        self.train_op, self.train_records = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule, return_summaries=False)

        # --- summaries ---

        output = network_tensors["output"]
        recorded_tensors.update({
            "loss_" + name: tf_mean_sum(builder(output, inp))
            for name, builder in loss_builders.items()
        })

        intersection = recorded_tensors.keys() & network_recorded_tensors.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)
        recorded_tensors.update(network_recorded_tensors)

        intersection = recorded_tensors.keys() & self.network.eval_funcs.keys()
        assert not intersection, "Key sets have non-zero intersection: {}".format(intersection)

        # For running functions, during evaluation, that are not implemented in tensorflow
        self.evaluator = Evaluator(self.network.eval_funcs, network_tensors, self)

        self.train_records_ph = {k: tf.placeholder(tf.float32, name=k + "_summary") for k in self.train_records}
        train_summaries = [tf.summary.scalar(k, t) for k, t in self.train_records_ph.items()]

        self.recorded_tensors_ph = {k: tf.placeholder(tf.float32, name=k + "_summary") for k in self.recorded_tensors}
        recorded_tensors_summaries = [tf.summary.scalar(k, t) for k, t in self.recorded_tensors_ph.items()]

        self.eval_funcs_ph = {k: tf.placeholder(tf.float32, name=k + "_summary") for k in self.network.eval_funcs}
        eval_funcs_summaries = [tf.summary.scalar(k, t) for k, t in self.eval_funcs_ph.items()]

        self.train_summary_op = tf.summary.merge(recorded_tensors_summaries + train_summaries)
        self.val_summary_op = tf.summary.merge(recorded_tensors_summaries + eval_funcs_summaries)
