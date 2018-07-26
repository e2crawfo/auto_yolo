import tensorflow as tf
import numpy as np
from sklearn.cluster import k_means
import collections

from dps import cfg
from dps.updater import Updater as _Updater
from dps.utils import Param, prime_factors
from dps.utils.tf import (
    ConvNet, build_gradient_train_op, tf_mean_sum,
    ScopedFunction, build_scheduled_value, MLP)
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


class Backbone(ConvNet):
    pixels_per_cell = Param()
    kernel_size = Param()
    n_channels = Param()
    n_final_layers = Param(2)

    def __init__(self, check_output_shape=False, **kwargs):
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


class InverseBackbone(ConvNet):
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

        super(InverseBackbone, self).__init__(layout, check_output_shape=False, **kwargs)


class NewBackbone(ConvNet):
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


class NextStep(ConvNet):
    kernel_size = Param()
    n_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME"),
            dict(filters=self.n_channels, kernel_size=self.kernel_size, strides=1, padding="SAME"),
        ]
        super(NextStep, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder(ConvNet):
    n_decoder_channels = Param()

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=5, strides=1, padding="VALID", transpose=True),
            dict(filters=self.n_decoder_channels, kernel_size=3, strides=2, padding="SAME", transpose=True),
            dict(filters=4, kernel_size=4, strides=1, padding="SAME", transpose=True),
        ]
        super(ObjectDecoder, self).__init__(layout, check_output_shape=True, **kwargs)


class ObjectDecoder28x28(ConvNet):
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


def compute_iou(box, others):
    # box: y_min, y_max, x_min, x_max, area
    # others: (n_boxes, 5)
    top = np.maximum(box[0], others[:, 0])
    bottom = np.minimum(box[1], others[:, 1])
    left = np.maximum(box[2], others[:, 2])
    right = np.minimum(box[3], others[:, 3])

    overlap_height = np.maximum(0., bottom - top)
    overlap_width = np.maximum(0., right - left)
    overlap_area = overlap_height * overlap_width

    return overlap_area / (box[4] + others[:, 4] - overlap_area)


def mAP(pred_boxes, gt_boxes, n_classes, recall_values=None, iou_threshold=None):
    """ Calculate mean average precision on a dataset.

    Averages over:
        classes, recall_values, iou_threshold

    pred_boxes: [[class, conf, y_min, y_max, x_min, x_max] * n_boxes] * n_images
    gt_boxes: [[class, y_min, y_max, x_min, x_max] * n_boxes] * n_images

    """
    if recall_values is None:
        recall_values = np.linspace(0.0, 1.0, 11)

    if iou_threshold is None:
        iou_threshold = np.linspace(0.5, 0.95, 10)

    ap = []

    for c in range(n_classes):
        _ap = []
        for iou_thresh in iou_threshold:
            predicted_list = []  # Each element is (confidence, ground-truth (0 or 1))
            n_positives = 0

            for pred, gt in zip(pred_boxes, gt_boxes):
                # Within a single image

                # Sort by decreasing confidence within current class.
                pred_c = sorted([b for cls, *b in pred if cls == c], key=lambda k: -k[0])
                area = [(ymax - ymin) * (xmax - xmin) for _, ymin, ymax, xmin, xmax in pred_c]
                pred_c = [(*b, a) for b, a in zip(pred_c, area)]

                gt_c = [b for cls, *b in gt if cls == c]
                n_positives += len(gt_c)

                if not gt_c:
                    predicted_list.extend((conf, 0) for conf, *b in pred_c)
                    continue

                gt_c = np.array(gt_c)
                gt_c_area = (gt_c[:, 1] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 2])
                gt_c = np.concatenate([gt_c, gt_c_area[..., None]], axis=1)

                for conf, *box in pred_c:
                    iou = compute_iou(box, gt_c)
                    best_idx = np.argmax(iou)
                    best_iou = iou[best_idx]
                    if best_iou > iou_thresh:
                        predicted_list.append((conf, 1.))
                        gt_c = np.delete(gt_c, best_idx, axis=0)
                    else:
                        predicted_list.append((conf, 0.))

                    if not gt_c.shape[0]:
                        break

            if not predicted_list:
                ap.append(0.0)
                continue

            # Sort predictions by decreasing confidence.
            predicted_list = np.array(sorted(predicted_list, key=lambda k: -k[0]), dtype=np.float32)

            # Compute AP
            cs = np.cumsum(predicted_list[:, 1])
            precision = cs / (np.arange(predicted_list.shape[0]) + 1)
            recall = cs / n_positives

            for r in recall_values:
                p = precision[recall >= r]
                _ap.append(0. if p.size == 0 else p.max())
        ap.append(np.mean(_ap))
    return np.mean(ap)


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

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape
        self.network = cfg.build_network(env, scope="network")

        super(Updater, self).__init__(env, scope=scope, **kwargs)

    @property
    def completion(self):
        return self.env.datasets['train'].completion

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
        if mode == "val":
            feed_dict = self.data_manager.do_val()
        elif mode == "test":
            feed_dict = self.data_manager.do_test()
        else:
            raise Exception("Unknown evaluation mode: {}".format(mode))

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
        self.data_manager = DataManager(self.env.datasets['train'],
                                        self.env.datasets['val'],
                                        self.env.datasets['test'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        inp, *labels = self.data_manager.iterator.get_next()

        if cfg.background_cfg.mode == "static":
            with cfg.background_cfg.static_cfg:
                from kmodes.kmodes import KModes
                print("Clustering...")
                print(cfg.background_cfg.static_cfg)

                cluster_data = self.env.datasets["train"].X
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


class VariationalAutoencoder(ScopedFunction):
    fixed_weights = Param()
    fixed_values = Param()
    no_gradient = Param()

    xent_loss = Param()

    attr_prior_mean = Param()
    attr_prior_std = Param()

    A = Param()

    train_reconstruction = Param()
    reconstruction_weight = Param()

    train_kl = Param()
    kl_weight = Param()

    train_math = Param()
    math_weight = Param()
    math_A = Param()

    noisy = Param()

    representation_network = None
    math_input_network = None
    math_network = None

    eval_funcs = dict()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        self.attr_prior_mean = build_scheduled_value(self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(self.attr_prior_std, "attr_prior_std")

        self.reconstruction_weight = build_scheduled_value(
            self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")
        self.math_weight = build_scheduled_value(self.math_weight, "math_weight")

        if not self.noisy and self.train_kl:
            raise Exception("If `noisy` is False, `train_kl` must also be False.")

        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

        if isinstance(self.no_gradient, str):
            self.no_gradient = self.no_gradient.split()

        super(VariationalAutoencoder, self).__init__(scope=scope, **kwargs)

    def build_math_representation(self):
        attr_shape = tf.shape(self._tensors['attr'])
        attr = tf.reshape(self._tensors['attr'], (-1, self.A))

        math_A = self.A if self.math_A is None else self.math_A
        math_attr = self.math_input_network(attr, math_A, self.is_training)

        new_shape = tf.concat([attr_shape[:-1], [math_A]], axis=0)
        math_attr = tf.reshape(math_attr, new_shape)
        self._tensors["math_attr"] = math_attr
        return math_attr

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

    def _process_labels(self, labels):
        self._tensors.update(
            annotations=labels[0],
            n_annotations=labels[1],
            targets=labels[2],
        )

    def _call(self, inp, _, is_training):
        self.original_inp = inp
        inp, labels, background = inp

        self._tensors = dict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            background=background,
            batch_size=tf.shape(inp)[0],
        )

        self.labels = labels
        self._process_labels(labels)

        self.recorded_tensors = dict(
            batch_size=tf.to_float(self.batch_size),
            float_is_training=self.float_is_training
        )

        self.losses = dict()

        with tf.variable_scope("representation", reuse=self.initialized):
            self.build_representation()

        if self.train_math:
            with tf.variable_scope("math", reuse=self.initialized):
                self.build_math()

        return dict(
            tensors=self._tensors,
            recorded_tensors=self.recorded_tensors,
            losses=self.losses,
        )

    def build_math(self):
        # --- init modules ---

        if self.math_input_network is None:
            self.math_input_network = cfg.build_math_input(scope="math_input_network")
            if "math" in self.fixed_weights:
                self.math_input_network.fix_variables()

        if self.math_network is None:
            self.math_network = cfg.build_math_network(scope="math_network")

            if "math" in self.fixed_weights:
                self.math_network.fix_variables()

        # --- process representation ---

        math_rep = self.build_math_representation()

        logits = self.math_network(math_rep, cfg.n_classes, self.is_training)

        # --- record values and losses ---

        self._tensors["prediction"] = tf.nn.softmax(logits)

        recorded_tensors = self.recorded_tensors

        if self.math_weight is not None:
            recorded_tensors["raw_loss_math"] = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._tensors["targets"],
                    logits=logits
                )
            )

            self.losses["math"] = self.math_weight * recorded_tensors["raw_loss_math"]

        recorded_tensors["math_accuracy"] = tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.argmax(logits, axis=1),
                    tf.argmax(self._tensors['targets'], axis=1)
                )
            )
        )

        recorded_tensors["math_1norm"] = tf.reduce_mean(
            tf.to_float(
                tf.abs(tf.argmax(logits, axis=1) - tf.argmax(self._tensors['targets'], axis=1))
            )
        )

        recorded_tensors["math_correct_prob"] = tf.reduce_mean(
            tf.reduce_sum(tf.nn.softmax(logits) * self._tensors['targets'], axis=1)
        )


class SimpleRecurrentRegressionNetwork(ScopedFunction):
    cell = None
    output_network = None

    def _call(self, inp, output_size, is_training):
        if self.cell is None:
            self.cell = cfg.build_math_cell(scope="regression_cell")
        if self.output_network is None:
            self.output_network = cfg.build_math_output(scope="math_output")

        batch_size = tf.shape(inp)[0]
        A = inp.shape[-1]
        _inp = tf.reshape(inp, (batch_size, -1, A))

        output, final_state = tf.nn.dynamic_rnn(
            self.cell, _inp, initial_state=self.cell.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False)

        return self.output_network(output[:, -1, :], output_size, is_training)


class SoftmaxMLP(MLP):
    def __init__(self, n_outputs, temp, n_units=None, scope=None, **fc_kwargs):
        self.n_outputs = n_outputs
        self.temp = temp

        super(SoftmaxMLP, self).__init__(scope=scope, n_units=n_units, **fc_kwargs)

    def _call(self, inp, output_size, is_training):
        output = super(SoftmaxMLP, self)._call(inp, output_size, is_training)
        return tf.nn.softmax(output / self.temp)


class SequentialRegressionNetwork(ScopedFunction):
    h_cell = None
    w_cell = None
    b_cell = None

    output_network = None

    def _call(self, _inp, output_size, is_training):
        if self.h_cell is None:
            self.h_cell = cfg.build_math_cell(scope="regression_h_cell")
            self.w_cell = cfg.build_math_cell(scope="regression_w_cell")
            self.b_cell = cfg.build_math_cell(scope="regression_b_cell")

        edge_state = self.h_cell.zero_state(tf.shape(_inp)[0], tf.float32)

        H, W, B = tuple(int(i) for i in _inp.shape[1:4])
        h_states = np.empty((H, W, B), dtype=np.object)
        w_states = np.empty((H, W, B), dtype=np.object)
        b_states = np.empty((H, W, B), dtype=np.object)

        for h in range(H):
            for w in range(W):
                for b in range(B):
                    h_state = h_states[h-1, w, b] if h > 0 else edge_state
                    w_state = w_states[h, w-1, b] if w > 0 else edge_state
                    b_state = b_states[h, w, b-1] if b > 0 else edge_state

                    inp = _inp[:, h, w, b, :]

                    h_inp = tf.concat([inp, w_state.h, b_state.h], axis=1)
                    _, h_states[h, w, b] = self.h_cell(h_inp, h_state)

                    w_inp = tf.concat([inp, h_state.h, b_state.h], axis=1)
                    _, w_states[h, w, b] = self.w_cell(w_inp, w_state)

                    b_inp = tf.concat([inp, h_state.h, w_state.h], axis=1)
                    _, b_states[h, w, b] = self.b_cell(b_inp, b_state)

        if self.output_network is None:
            self.output_network = cfg.build_math_output(scope="math_output")

        final_layer_input = tf.concat(
            [h_states[-1, -1, -1].h,
             w_states[-1, -1, -1].h,
             b_states[-1, -1, -1].h],
            axis=1)

        return self.output_network(final_layer_input, output_size, is_training)


class ObjectBasedRegressionNetwork(ScopedFunction):
    n_objects = Param(5)

    embedding = None
    output_network = None

    def _call(self, _inp, output_size, is_training):
        batch_size = tf.shape(_inp)[0]
        H, W, B, A = tuple(int(i) for i in _inp.shape[1:])

        if self.embedding is None:
            self.embedding = tf.get_variable(
                "embedding", shape=(int(A/2), self.n_objects), dtype=tf.float32)

        inp = tf.reshape(_inp, (batch_size, H * W * B, A))
        key, value = tf.split(inp, 2, axis=2)
        raw_attention = tf.tensordot(key, self.embedding, [[2], [0]])
        attention = tf.nn.softmax(raw_attention, axis=1)

        attention_t = tf.transpose(attention, (0, 2, 1))
        weighted_value = tf.matmul(attention_t, value)

        flat_weighted_value = tf.reshape(
            weighted_value, (batch_size, self.n_objects * int(A/2)))

        if self.output_network is None:
            self.output_network = cfg.build_math_output(scope="math_output")

        return self.output_network(flat_weighted_value, output_size, is_training)


class ConvolutionalRegressionNetwork(ScopedFunction):
    network = None

    def _call(self, inp, output_size, is_training):
        if self.network is None:
            self.network = cfg.build_convolutional_network(scope="regression_network")

        return self.network(inp['attr'], output_size, is_training)


class AttentionRegressionNetwork(ConvNet):
    ar_n_filters = Param(128)

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.ar_n_filters, kernel_size=3, padding="SAME", strides=1),
            dict(filters=self.ar_n_filters, kernel_size=3, padding="SAME", strides=1),
            dict(filters=4, kernel_size=1, padding="SAME", strides=1),
        ]
        super(AttentionRegressionNetwork, self).__init__(
            layout, check_output_shape=False, **kwargs)

    def _call(self, inp, output_size, is_training):
        self.layout[-1]['filters'] = output_size + 1

        batch_size = tf.shape(inp)[0]
        inp = tf.reshape(
            inp, (batch_size, *inp.shape[1:3], inp.shape[3] * inp.shape[4]))
        output = super(AttentionRegressionNetwork, self)._call(inp, output_size, is_training)
        output = tf.reshape(
            output, (batch_size, output.shape[1] * output.shape[2], output.shape[3]))

        logits, attention = tf.split(output, [output_size, 1], axis=2)

        attention = tf.nn.softmax(attention, axis=1)
        weighted_output = logits * attention

        return tf.reduce_sum(weighted_output, axis=1)


class AverageRegressionNetwork(ConvNet):
    """ Run a conv-net and then global mean pooling. """
    ar_n_filters = Param(128)

    def __init__(self, **kwargs):
        layout = [
            dict(filters=self.ar_n_filters, kernel_size=3, padding="SAME", strides=1),
            dict(filters=self.ar_n_filters, kernel_size=3, padding="SAME", strides=1),
            dict(filters=4, kernel_size=1, padding="SAME", strides=1),
        ]
        super(AttentionRegressionNetwork, self).__init__(layout, check_output_shape=False, **kwargs)

    def _call(self, inp, output_size, is_training):
        self.layout[-1]['filters'] = output_size

        batch_size = tf.shape(inp)[0]
        inp = tf.reshape(inp, (batch_size, *inp.shape[1:3], inp.shape[3] * inp.shape[4]))
        output = super(AttentionRegressionNetwork, self)._call(inp, output_size, is_training)
        return tf.reduce_mean(output, axis=(1, 2))


class RelationNetwork(ScopedFunction):
    f = None
    g = None

    f_dim = Param(100)

    def _call(self, inp, output_size, is_training):
        # Assumes objects range of all but the first and last dimensions
        batch_size = tf.shape(inp)[0]
        spatial_shape = inp.shape[1:-1]
        n_objects = np.prod(spatial_shape)
        obj_dim = inp.shape[-1]
        inp = tf.reshape(inp, (batch_size, n_objects, obj_dim))

        if self.f is None:
            self.f = cfg.build_relation_network_f(scope="relation_network_f")

        if self.g is None:
            self.g = cfg.build_relation_network_g(scope="relation_network_g")

        f_inputs = []
        for i in range(n_objects):
            for j in range(n_objects):
                f_inputs.append(tf.concat([inp[:, i, :], inp[:, j, :]], axis=1))
        f_inputs = tf.concat(f_inputs, axis=0)

        f_output = self.f(f_inputs, self.f_dim, is_training)
        f_output = tf.split(f_output, n_objects**2, axis=0)

        g_input = tf.concat(f_output, axis=1)
        return self.g(g_input, output_size, is_training)


def addition(left, right):
    m = left.shape[1]
    n = right.shape[1]

    mat = tf.to_float(
        tf.equal(
            tf.reshape(tf.range(m)[:, None] + tf.range(n)[None, :], (-1, 1)),
            tf.range(m + n - 1)[None, :]))

    outer_product = tf.matmul(left[:, :, None], right[:, None, :])
    outer_product = tf.reshape(outer_product, (-1, m * n))

    return tf.tensordot(outer_product, mat)


def addition_compact(left, right):
    # Runtime is O((m+n) * n), so smaller value should be put second.
    batch_size = tf.shape(left)[0]
    m = left.shape[1]
    n = right.shape[1]

    running_sum = tf.zeros((batch_size, m+n-1))
    to_add = tf.concat([left, tf.zeros((batch_size, n-1))], axis=1)

    for i in range(n):
        running_sum += to_add * right[:, i:i+1]
        to_add = tf.manip.roll(to_add, shift=1, axis=1)
    return running_sum


def addition_compact_logspace(left, right):
    # Runtime is O((m+n) * n), so smaller value should be put second.
    batch_size = tf.shape(left)[0]
    n = right.shape[1]

    tensors = []
    to_add = tf.concat([left, -100 * tf.ones((batch_size, n-1))], axis=1)

    for i in range(n):
        tensors.append(to_add + right[:, i:i+1])
        to_add = tf.manip.roll(to_add, shift=1, axis=1)
    return tf.reduce_logsumexp(tf.stack(tensors, axis=2), axis=2)


class AdditionNetwork(ScopedFunction):
    def _call(self, inp, output_size, is_training):
        H, W, B, _ = tuple(int(i) for i in inp.shape[1:])

        # inp = tf.log(tf.nn.softmax(tf.clip_by_value(inp, -10., 10.), axis=4))
        inp = inp - tf.reduce_logsumexp(inp, axis=4, keepdims=True)

        running_sum = inp[:, 0, 0, 0, :]

        for h in range(H):
            for w in range(W):
                for b in range(B):
                    if h == 0 and w == 0 and b == 0:
                        pass
                    else:
                        right = inp[:, h, w, b, :]
                        running_sum = addition_compact_logspace(running_sum, right)

        assert running_sum.shape[1] == output_size
        return running_sum
