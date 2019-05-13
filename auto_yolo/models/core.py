import tensorflow as tf
import numpy as np
import collections
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib import animation

from dps import cfg
from dps.updater import Updater as _Updater
from dps.utils import Param, square_subplots
from dps.utils.tf import (
    build_gradient_train_op, apply_mask_and_group_at_front,
    ScopedFunction, build_scheduled_value, FIXED_COLLECTION)
from dps.updater import DataManager
from dps.train import Hook


def normal_kl(mean, std, prior_mean, prior_std):
    var = std**2
    prior_var = prior_std**2

    return 0.5 * (
        tf.log(prior_var) - tf.log(var)
        - 1.0 + var / prior_var
        + tf.square(mean - prior_mean) / prior_var
    )


def normal_vae(mean, std, prior_mean, prior_std):
    sample = mean + tf.random_normal(tf.shape(mean)) * std
    kl = normal_kl(mean, std, prior_mean, prior_std)
    return sample, kl


def concrete_binary_pre_sigmoid_sample(log_odds, temperature, eps=10e-10):
    u = tf.random_uniform(tf.shape(log_odds), minval=0, maxval=1)
    noise = tf.log(u + eps) - tf.log(1.0 - u + eps)
    return (log_odds + noise) / temperature


def concrete_binary_sample_kl(posterior_pre_sigmoid_sample,
                              posterior_log_odds, posterior_temperature,
                              prior_log_odds, prior_temperature,
                              eps=10e-10):
    """ Compute KL divergence between two RelaxedBernoulli distributions. Rather than compute it
        directly, we compute the KL divergence between two Logistic distributions (just before the sigmoid),
        as this is supposed to avoid numerical issues. `posterior_pre_sigmoid_samples` must be a sample from the posterior
        Logistic distribution, used to form the sample-based estimate of the KL.

        This computes KL(posterior || prior).

    https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/RelaxedBernoulli

    """
    y = posterior_pre_sigmoid_sample

    y_times_prior_temp = y * prior_temperature
    log_prior = (
        tf.log(prior_temperature + eps)
        - y_times_prior_temp
        + prior_log_odds
        - 2.0 * tf.log(1.0 + tf.exp(-y_times_prior_temp + prior_log_odds) + eps)
    )

    y_times_posterior_temp = y * posterior_temperature
    log_posterior = (
        tf.log(posterior_temperature + eps)
        - y_times_posterior_temp
        + posterior_log_odds
        - 2.0 * tf.log(1.0 + tf.exp(-y_times_posterior_temp + posterior_log_odds) + eps)
    )

    return log_posterior - log_prior


def tf_safe_log(value, replacement_value=-100.0):
    log_value = tf.log(value + 1e-9)
    replace = tf.logical_or(tf.is_nan(log_value), tf.is_inf(log_value))
    log_value = tf.where(replace, replacement_value * tf.ones_like(log_value), log_value)
    return log_value


def np_safe_log(value, replacement_value=-100.0):
    log_value = np.log(value + 1e-9)
    replace = np.logical_or(np.isnan(log_value), np.isinf(log_value))
    log_value = np.where(replace, replacement_value * np.ones_like(log_value), log_value)
    return log_value.astype(value.dtype)


def xent_loss(*, pred, label, tf=True):
    if tf:
        return -(label * tf_safe_log(pred) + (1. - label) * tf_safe_log(1. - pred))
    else:
        return -(label * np_safe_log(pred) + (1. - label) * np_safe_log(1. - pred))


class Evaluator(object):
    """ A helper object for running a list of functions on a collection of evaluated tensors.

    Parameters
    ----------
    functions: a list of functions, each with an attribute `keys_accessed`
               listing the keys required by that function
    tensors: a (possibly nested) dictionary of tensors which will provide the input to the functions
    updater: the updater object, passed into the functions at eval time

    """
    def __init__(self, functions, tensors, updater):
        self.functions = functions
        self.updater = updater

        if not functions:
            self.fetches = {}
            return

        fetch_keys = set()
        for f in functions.values():
            keys_accessed = f.keys_accessed
            if isinstance(keys_accessed, str):
                keys_accessed = keys_accessed.split()
            for key in keys_accessed:
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
        """ fetched should be a dictionary containing numpy arrays derived by fetching the tensors
            in self.fetches """
        record = {}
        for name, func in self.functions.items():
            result = func(fetched, self.updater)

            if isinstance(result, dict):
                for k, v in result.items():
                    record["{}:{}".format(name, k)] = np.mean(v)
            else:
                record[name] = np.mean(result)

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
            predicted_list = []  # Each element is of the form (confidence, ground-truth (0 or 1))
            n_positives_gt = 0

            for pred, gt in zip(pred_boxes, gt_boxes):
                # Within a single image

                # Sort by decreasing confidence within current class.
                pred_c = sorted([b for cls, *b in pred if cls == c], key=lambda k: -k[0])
                area = [(ymax - ymin) * (xmax - xmin) for _, ymin, ymax, xmin, xmax in pred_c]
                pred_c = [(*b, a) for b, a in zip(pred_c, area)]

                gt_c = [b for cls, *b in gt if cls == c]
                n_positives_gt += len(gt_c)

                if not gt_c:
                    predicted_list.extend((conf, 0) for conf, *_ in pred_c)
                    continue

                gt_c = np.array(gt_c)
                gt_c_area = (gt_c[:, 1] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 2])
                gt_c = np.concatenate([gt_c, gt_c_area[..., None]], axis=1)

                used = [0] * len(gt_c)

                for conf, *box in pred_c:
                    iou = compute_iou(box, gt_c)
                    best_idx = np.argmax(iou)
                    best_iou = iou[best_idx]

                    if best_iou > iou_thresh and not used[best_idx]:
                        predicted_list.append((conf, 1.))
                        used[best_idx] = 1
                    else:
                        predicted_list.append((conf, 0.))

            if not predicted_list:
                ap.append(0.0)
                continue

            # Sort predictions by decreasing confidence.
            predicted_list = np.array(sorted(predicted_list, key=lambda k: -k[0]), dtype=np.float32)

            # Compute AP
            cs = np.cumsum(predicted_list[:, 1])
            precision = cs / (np.arange(predicted_list.shape[0]) + 1)
            recall = cs / n_positives_gt

            for r in recall_values:
                p = precision[recall >= r]
                _ap.append(0. if p.size == 0 else p.max())

        ap.append(np.mean(_ap) if _ap else 0.0)
    return np.mean(ap)


class AP:
    keys_accessed = "normalized_box obj annotations n_annotations"

    def __init__(self, iou_threshold=None):
        if iou_threshold is not None:
            try:
                iou_threshold = list(iou_threshold)
            except (TypeError, ValueError):
                iou_threshold = [float(iou_threshold)]
        self.iou_threshold = iou_threshold

    def _process_data(self, tensors, updater):
        obj = tensors['obj']
        top, left, height, width = np.split(tensors['normalized_box'], 4, axis=-1)

        batch_size = obj.shape[0]
        n_frames = getattr(updater.network, 'n_frames', 0)

        annotations = tensors["annotations"]
        n_annotations = tensors["n_annotations"]

        if n_frames > 0:
            n_objects = np.prod(obj.shape[2:-1])
        else:
            n_objects = np.prod(obj.shape[1:-1])
            annotations = annotations.reshape(batch_size, 1, *annotations.shape[1:])
            n_frames = 1

        shape = (batch_size, n_frames, n_objects)

        obj = obj.reshape(*shape)
        n_digits = n_objects * np.ones((batch_size, n_frames), dtype=np.int32)
        top = updater.network.image_height * top.reshape(*shape)
        left = updater.network.image_width * left.reshape(*shape)
        height = updater.network.image_height * height.reshape(*shape)
        width = updater.network.image_width * width.reshape(*shape)

        return obj, n_digits, top, left, height, width, annotations, n_annotations

    def __call__(self, tensors, updater):
        obj, n_digits, top, left, height, width, annotations, n_annotations = self._process_data(tensors, updater)

        bottom = top + height
        right = left + width

        batch_size, n_frames = n_digits.shape[:2]

        ground_truth_boxes = []
        predicted_boxes = []

        for b in range(batch_size):
            for f in range(n_frames):
                _ground_truth_boxes = [
                    [0, *bbox]
                    for (valid, _, _, *bbox), _
                    in zip(annotations[b, f], range(n_annotations[b]))
                    if valid > 0.5]
                ground_truth_boxes.append(_ground_truth_boxes)

                _predicted_boxes = []
                for j in range(int(n_digits[b, f])):
                    o = obj[b, f, j]

                    if o > 0.0:
                        _predicted_boxes.append(
                            [0, o,
                             top[b, f, j],
                             bottom[b, f, j],
                             left[b, f, j],
                             right[b, f, j]])

                predicted_boxes.append(_predicted_boxes)

        return mAP(
            predicted_boxes, ground_truth_boxes, n_classes=1,
            iou_threshold=self.iou_threshold)


class Updater(_Updater):
    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.obs_shape
        *other, self.image_height, self.image_width, self.image_depth = self.obs_shape
        self.n_frames = other[0] if other else 0
        self.network = cfg.build_network(env, self, scope="network")

        super(Updater, self).__init__(env, scope=scope, **kwargs)

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

    def _build_graph(self):
        self.data_manager = DataManager(self.env.datasets['train'],
                                        self.env.datasets['val'],
                                        self.env.datasets['test'],
                                        cfg.batch_size)
        self.data_manager.build_graph()

        data = self.data_manager.iterator.get_next()
        self.inp = data["image"]
        network_outputs = self.network(data, self.data_manager.is_training)

        network_tensors = network_outputs["tensors"]
        network_recorded_tensors = network_outputs["recorded_tensors"]
        network_losses = network_outputs["losses"]

        self.tensors = network_tensors

        self.recorded_tensors = recorded_tensors = dict(global_step=tf.train.get_or_create_global_step())

        # --- loss ---

        self.loss = 0.0
        for name, tensor in network_losses.items():
            self.loss += tensor
            recorded_tensors['loss_' + name] = tensor
        recorded_tensors['loss'] = self.loss

        # --- train op ---

        tvars = self.trainable_variables(for_opt=True)

        self.train_op, self.train_records = build_gradient_train_op(
            self.loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        # --- recorded values ---

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

        data = self.data_manager.iterator.get_next()  # a dict mapping from names to tensors
        self.inp = data["image"]
        network_outputs = self.network(data, self.data_manager.is_training)

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


class TensorRecorder(ScopedFunction):
    _recorded_tensors = None

    def record_tensors(self, **kwargs):
        for k, v in kwargs.items():
            self.recorded_tensors[k] = tf.reduce_mean(tf.to_float(v))

    @property
    def recorded_tensors(self):
        if self._recorded_tensors is None:
            self._recorded_tensors = {}
        return self._recorded_tensors


class VariationalAutoencoder(TensorRecorder):
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
    max_possible_objects = Param()

    needs_background = True

    representation_network = None
    math_input_network = None
    math_network = None

    eval_funcs = dict()

    background_encoder = None
    background_decoder = None

    def __init__(self, env, updater, scope=None, **kwargs):
        self.updater = updater

        self.obs_shape = env.obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.attr_prior_mean = build_scheduled_value(self.attr_prior_mean, "attr_prior_mean")
        self.attr_prior_std = build_scheduled_value(self.attr_prior_std, "attr_prior_std")

        self.reconstruction_weight = build_scheduled_value(
            self.reconstruction_weight, "reconstruction_weight")
        self.kl_weight = build_scheduled_value(self.kl_weight, "kl_weight")
        self.math_weight = build_scheduled_value(self.math_weight, "math_weight")

        if not self.noisy and self.train_kl:
            raise Exception("If `noisy` is False, `train_kl` must also be False.")

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

    def _call(self, data, is_training):
        inp = data["image"]

        self._tensors = dict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            batch_size=tf.shape(inp)[0],
        )

        if "annotations" in data:
            self._tensors.update(
                annotations=data["annotations"]["data"],
                n_annotations=data["annotations"]["shapes"][:, 0],
                n_valid_annotations=tf.to_int32(
                    tf.reduce_sum(
                        data["annotations"]["data"][:, :, 0]
                        * tf.to_float(data["annotations"]["mask"][:, :, 0]),
                        axis=1
                    )
                )
            )

        if "label" in data:
            self._tensors.update(
                targets=data["label"],
            )

        if "background" in data:
            self._tensors.update(
                background=data["background"],
            )

        self.record_tensors(
            batch_size=self.batch_size,
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
            if cfg.background_cfg.mode == "colour":
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

                if len(background.shape) == 2:
                    # background decoder predicts a solid colour
                    assert background.shape[1] == 3

                    background = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))
                    background = background[:, None, None, :]
                    background = tf.tile(background, (1, self.inp.shape[1], self.inp.shape[2], 1))
                else:
                    background = background[:, :self.inp.shape[1], :self.inp.shape[2], :]
                    background = tf.nn.sigmoid(tf.clip_by_value(background, -10, 10))

            elif cfg.background_cfg.mode == "data":
                background = self._tensors["background"]

            else:
                raise Exception("Unrecognized background mode: {}.".format(cfg.background_cfg.mode))

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

        if self.max_possible_objects is not None:
            math_rep, _, mask = apply_mask_and_group_at_front(math_rep, mask)
            n_pad = self.max_possible_objects - tf.shape(math_rep)[1]
            mask = tf.cast(mask, tf.float32)

            batch_size = tf.shape(math_rep)[0]
            A = math_rep.shape[2]

            math_rep = tf.pad(math_rep, [(0, 0), (0, n_pad), (0, 0)])
            math_rep = tf.reshape(math_rep, (batch_size, self.max_possible_objects, A))

            mask = tf.pad(mask, [(0, 0), (0, n_pad)])
            mask = tf.reshape(mask, (batch_size, self.max_possible_objects, 1))

        mask_shape = tf.concat([tf.shape(math_rep)[:-1], [1]], axis=0)
        mask = tf.reshape(mask, mask_shape)

        math_rep = tf.concat([mask, math_rep], axis=-1)

        logits = self.math_network(math_rep, cfg.n_classes, self.is_training)

        # --- record values and losses ---

        self._tensors["prediction"] = tf.nn.softmax(logits)

        recorded_tensors = self.recorded_tensors

        if self.math_weight is not None:
            self.record_tensors(
                raw_loss_math=tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._tensors["targets"],
                    logits=logits
                )
            )

            self.losses["math"] = self.math_weight * recorded_tensors["raw_loss_math"]

        self.record_tensors(
            math_accuracy=tf.equal(
                tf.argmax(logits, axis=1),
                tf.argmax(self._tensors['targets'], axis=1)
            ),
            math_1norm=tf.abs(tf.argmax(logits, axis=1) - tf.argmax(self._tensors['targets'], axis=1)),
            math_correct_prob=tf.reduce_sum(tf.nn.softmax(logits) * self._tensors['targets'], axis=1)
        )
