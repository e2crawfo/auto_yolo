import tensorflow as tf
import numpy as np
from sklearn.cluster import k_means
import collections
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib import animation

from dps import cfg
from dps.updater import Updater as _Updater
from dps.utils import Param, prime_factors, square_subplots
from dps.utils.tf import (
    ConvNet, build_gradient_train_op, tf_mean_sum,
    ScopedFunction, build_scheduled_value, FIXED_COLLECTION)
from dps.updater import DataManager
from dps.train import Hook


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
        self.network = cfg.build_network(env, self, scope="network")

        super(Updater, self).__init__(env, scope=scope, **kwargs)

    @property
    def completion(self):
        return self.env.datasets['train'].completion

    def trainable_variables(self, for_opt):
        return self.network.trainable_variables(for_opt)

    def _update(self, batch_size):
        feed_dict = self.data_manager.do_train()

        sess = tf.get_default_session()
        _, record, train_record = sess.run(
            [self.train_op, self.recorded_tensors, self.train_records], feed_dict=feed_dict)
        record.update(train_record)

        return dict(train=record)

    def _evaluate(self, _batch_size, mode):
        if mode == "val":
            feed_dict = self.data_manager.do_val()
        elif mode == "test":
            feed_dict = self.data_manager.do_test()
        else:
            raise Exception("Unknown evaluation mode: {}".format(mode))

        record = collections.defaultdict(float)
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

            batch_size = _record['batch_size']

            for k, v in _record.items():
                record[k] += batch_size * v

            n_points += batch_size

        return {k: v / n_points for k, v in record.items()}

    def compute_validation_pixelwise_mean(self):
        sess = tf.get_default_session()

        mean = np.array([0, 0, 0])
        mean = None
        n_points = 0
        feed_dict = self.data_manager.do_val()

        while True:
            try:
                inp = sess.run(self.network.inp, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

            n_new = inp.shape[0]
            if mean is None:
                mean = np.mean(inp, axis=0)
            else:
                mean = mean * (n_points / (n_points + n_new)) + np.sum(inp, axis=0) / (n_points + n_new)
            n_points += n_new
        return mean

    def compute_validation_mean(self):
        sess = tf.get_default_session()

        mean = np.array([0, 0, 0])
        n_points = 0
        feed_dict = self.data_manager.do_val()

        while True:
            try:
                inp = sess.run(self.network.inp, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

            n_new = np.product(inp.shape[:3])
            mean = mean * (n_points / (n_points + n_new)) + np.sum(inp, axis=(0, 1, 2)) / (n_points + n_new)
            n_points += n_new
        return mean

    def compute_validation_mode(self):
        sess = tf.get_default_session()

        feed_dict = self.data_manager.do_val()
        counter = collections.Counter()

        while True:
            try:
                inp = sess.run(self.network.inp, feed_dict=feed_dict)
            except tf.errors.OutOfRangeError:
                break

            transformed = (255. * inp).astype('i')
            counter.update(tuple(t) for t in transformed.reshape(-1, 3))

        most_common = counter.most_common(1)[0][0]

        return np.array(most_common) / 255.

    def _build_graph(self):
        self.data_manager = DataManager(self.env.datasets['train'],
                                        self.env.datasets['val'],
                                        self.env.datasets['test'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        inp, *labels = self.data_manager.iterator.get_next()
        self.inp = inp
        network_outputs = self.network((inp, labels), self.data_manager.is_training)

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
            self.max_grad_norm, self.noise_schedule)

        # --- recorded values ---

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


class EvalHook(Hook):
    def __init__(self, dataset_class, plot_step=None, dataset_kwargs=None, **kwargs):
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs or {}
        self.dataset_kwargs['n_examples'] = cfg.n_val
        kwarg_string = "_".join("{}={}".format(k, v) for k, v in self.dataset_kwargs.items())
        name = dataset_class.__name__ + ("_" + kwarg_string if kwarg_string else "")
        self.name = name.replace(" ", "_")
        self.plot_step = plot_step
        super(EvalHook, self).__init__(final=True, **kwargs)

    def start_stage(self, training_loop, updater, stage_idx):
        # similar to `build_graph`

        self.network = updater.network

        dataset = self.dataset_class(**self.dataset_kwargs)
        self.data_manager = DataManager(val_dataset=dataset, batch_size=cfg.batch_size)
        self.data_manager.build_graph()

        inp, *labels = self.data_manager.iterator.get_next()
        self.inp = inp
        network_outputs = self.network((inp, labels), self.data_manager.is_training)

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

        # --- recorded values ---

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

    def step(self, training_loop, updater, step_idx=None):
        feed_dict = self.data_manager.do_val()
        record = collections.defaultdict(float)
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

            batch_size = _record['batch_size']

            for k, v in _record.items():
                record[k] += batch_size * v

            n_points += batch_size

        return {self.name: {k: v / n_points for k, v in record.items()}}

    def _plot(self, updater, rollouts):
        plt.ion()

        if updater.dataset.gym_dataset.image_obs:
            obs = rollouts.obs
        else:
            obs = rollouts.image

        fig, axes = square_subplots(rollouts.batch_size, figsize=(5, 5))
        plt.subplots_adjust(top=0.95, bottom=0, left=0, right=1, wspace=0.1, hspace=0.1)

        images = []
        for i, ax in enumerate(axes.flatten()):
            ax.set_aspect("equal")
            ax.set_axis_off()
            image = ax.imshow(np.zeros(obs.shape[2:]))
            images.append(image)

        def animate(t):
            for i in range(rollouts.batch_size):
                images[i].set_array(obs[t, i, :, :, :])

        anim = animation.FuncAnimation(fig, animate, frames=len(rollouts), interval=500)

        path = updater.exp_dir.path_for('plots', '{}_animation.gif'.format(self.name))
        anim.save(path, writer='imagemagick')

        plt.close(fig)


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

    needs_background = True

    representation_network = None
    math_input_network = None
    math_network = None

    eval_funcs = dict()

    background_encoder = None
    background_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        self.updater = updater

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
        return math_attr, self._tensors['obj']

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

        if len(labels) > 3:
            self._tensors.update(
                background=labels[3]
            )

    def _call(self, inp, is_training):
        self.original_inp = inp
        inp, labels = inp

        self._tensors = dict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
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
            if self.needs_background:
                self.build_background()
            self.build_representation()

        if self.train_math:
            with tf.variable_scope("math", reuse=self.initialized):
                self.build_math()

        return dict(
            tensors=self._tensors,
            recorded_tensors=self.recorded_tensors,
            losses=self.losses,
        )

    def build_background(self):
        if self.needs_background:
            if cfg.background_cfg.mode == "static":
                raise Exception("NotImplemented")

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
                    centroids = np.clip(centroids, 0.0, 1.0)
                    centroids = centroids.reshape(cfg.n_clusters, *image_shape)

            elif cfg.background_cfg.mode == "pixelwise_mean":
                mean = self.updater.compute_validation_pixelwise_mean()
                background = mean[None, :, :, :] * tf.ones_like(self.inp)

            elif cfg.background_cfg.mode == "mean":
                mean = self.updater.compute_validation_mean()
                background = mean[None, None, None, :] * tf.ones_like(self.inp)

            elif cfg.background_cfg.mode == "mode":
                mode = self.updater.compute_validation_mode()
                background = mode[None, None, None, :] * tf.ones_like(self.inp)

            elif cfg.background_cfg.mode == "colour":
                rgb = np.array(to_rgb(cfg.background_cfg.colour))[None, None, None, :]
                background = rgb * tf.ones_like(self.inp)

            elif cfg.background_cfg.mode == "learn_solid":
                # Learn a solid colour for the background
                self.solid_background_logits = tf.get_variable("solid_background", initializer=[0.0, 0.0, 0.0])
                if "background" in self.fixed_weights:
                    tf.add_to_collection(FIXED_COLLECTION, self.solid_background_logits)
                solid_background = tf.nn.sigmoid(10 * self.solid_background_logits)
                background = solid_background[None, None, None, :] * tf.ones_like(self.inp)

            elif cfg.background_cfg.mode == "learn":
                if self.background_encoder is None:
                    self.background_encoder = cfg.build_background_encoder(scope="background_encoder")
                    if "background_encoder" in self.fixed_weights:
                        self.background_encoder.fix_variables()

                if self.background_decoder is None:
                    self.background_decoder = cfg.build_background_decoder(scope="background_decoder")
                    if "background_decoder" in self.fixed_weights:
                        self.background_decoder.fix_variables()

                bg_attr = self.background_encoder(self.inp, 2 * cfg.background_cfg.A, self.is_training)
                bg_attr_mean, bg_attr_log_std = tf.split(bg_attr, 2, axis=-1)
                bg_attr_std = tf.exp(bg_attr_log_std)
                if not self.noisy:
                    bg_attr_std = tf.zeros_like(bg_attr_std)

                bg_attr, bg_attr_kl = normal_vae(bg_attr_mean, bg_attr_std, self.attr_prior_mean, self.attr_prior_std)

                self._tensors.update(
                    bg_attr_mean=bg_attr_mean,
                    bg_attr_std=bg_attr_std,
                    bg_attr_kl=bg_attr_kl,
                    bg_attr=bg_attr)

                # --- decode ---

                background = self.background_decoder(bg_attr, self.inp.shape[1:], self.is_training)
                background = background[:, :self.inp.shape[1], :self.inp.shape[2], :]
                background = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))

            elif cfg.background_cfg.mode == "data":
                background = self._tensors["background"]
            else:
                background = tf.zeros_like(self.inp)

            self._tensors["background"] = background

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

        math_rep, mask = self.build_math_representation()
        mask_shape = tf.concat([tf.shape(math_rep)[:-1], [1]], axis=0)
        mask = tf.reshape(mask, mask_shape)
        math_rep = tf.concat([mask, math_rep], axis=-1)

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
