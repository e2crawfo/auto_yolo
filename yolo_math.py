import tensorflow as tf
from tensorflow.contrib.slim import fully_connected
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

from dps import cfg
from dps.utils import Param, Parameterized, Config
from dps.utils.tf import (
    trainable_variables, ScopedFunction, MLP, FullyConvolutional, tf_mean_sum)
from dps.env.advanced import yolo_rl, yolo_air
from dps.datasets import VisualArithmeticDataset


class Env(object):
    def __init__(self):
        train = VisualArithmeticDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = VisualArithmeticDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


class SequentialRegressionNetwork(ScopedFunction):
    h_cell = None
    w_cell = None
    b_cell = None

    output_network = None

    def _call(self, _inp, output_size, is_training):
        """ program is the program dictionary from YoloAir_Network """

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
        """ program is the program dictionary from YoloAir_Network """

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

        flat_weighted_value = tf.reshape(weighted_value, (batch_size, self.n_objects * int(A/2)))

        if self.output_network is None:
            self.output_network = cfg.build_math_output(scope="math_output")

        return self.output_network(flat_weighted_value, output_size, is_training)


class ConvolutionalRegressionNetwork(ScopedFunction):
    network = None

    def _call(self, inp, output_size, is_training):
        """ inp is the program dictionary from YoloAir_Network """

        if self.network is None:
            self.network = cfg.build_convolutional_network(scope="regression_network")

        return self.network(inp['attr'], output_size, is_training)


class YoloAir_MathNetwork(yolo_air.YoloAir_Network):
    math_weight = Param()
    largest_digit = Param()

    math_input_network = None
    math_network = None

    @property
    def n_classes(self):
        return self.largest_digit + 1

    def trainable_variables(self, for_opt):
        tvars = super(YoloAir_MathNetwork, self).trainable_variables(for_opt)
        math_network_tvars = trainable_variables(self.math_network.scope, for_opt=for_opt)
        tvars.extend(math_network_tvars)
        math_input_network_tvars = trainable_variables(self.math_input_network.scope, for_opt=for_opt)
        tvars.extend(math_input_network_tvars)
        return tvars

    def build_math_representation(self, math_attr):
        # Use raw_obj so that there is no discrepancy between validation and train
        return self._tensors["raw_obj"] * math_attr

    def build_graph(self, *args, **kwargs):
        result = super(YoloAir_MathNetwork, self).build_graph(*args, **kwargs)

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


class ConvNet(FullyConvolutional):
    def __init__(self):
        layout = [
            dict(filters=64, kernel_size=5, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=2, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=1, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=2, padding="VALID"),
            dict(filters=128, kernel_size=5, strides=1, padding="VALID"),
        ]
        super(ConvNet, self).__init__(layout, check_output_shape=False)

    def _call(self, inp, output_size, is_training):
        output = super(ConvNet, self)._call(inp, output_size, is_training)
        output = tf.nn.relu(output)  # FullyConvolutional doesn't apply non-linearity to final layer
        size = np.product([int(i) for i in output.shape[1:]])
        output = tf.reshape(output, (tf.shape(inp)[0], size))
        output = fully_connected(output, 100)
        output = fully_connected(output, output_size, activation_fn=None)
        return output


class SimpleMathNetwork(Parameterized):
    largest_digit = Param()
    A = Param()
    pixels_per_cell = Param()
    fixed_weights = Param("")
    train_reconstruction = Param(True)
    train_kl = Param(True)
    variational = Param(True)
    math_weight = Param(1.0)
    xent_loss = Param(True)
    code_prior_mean = Param(0.0)
    code_prior_std = Param(1.0)

    encoder = None
    decoder = None
    math_input_network = None
    math_network = None

    def __init__(self, env, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.H = int(np.ceil(self.image_height / self.pixels_per_cell[0]))
        self.W = int(np.ceil(self.image_width / self.pixels_per_cell[1]))
        self.HW = self.H * self.W
        self.eval_funcs = dict()
        if isinstance(self.fixed_weights, str):
            self.fixed_weights = self.fixed_weights.split()

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

    def trainable_variables(self, for_opt):

        tvars = []
        for sf in [self.encoder, self.decoder, self.math_input_network, self.math_network]:
            tvars.extend(trainable_variables(sf.scope, for_opt=for_opt))

        return tvars

    def _process_labels(self, labels):
        self._tensors.update(
            annotations=labels[0],
            n_annotations=labels[1],
            targets=labels[2],
        )

    def build_math_representation(self, math_code):
        return math_code

    @property
    def n_classes(self):
        return self.largest_digit + 1

    def build_graph(self, inp, labels, is_training, background):
        attr_dim = 2 * self.A if self.variational else self.A

        # --- init modules ---

        if self.encoder is None:
            self.encoder = cfg.build_math_encoder(scope="math_encoder")
            if "encoder" in self.fixed_weights:
                self.encoder.fix_variables()
            self.encoder.layout[-1]['filters'] = attr_dim

        if self.decoder is None:
            self.decoder = cfg.build_math_decoder(scope="math_decoder")
            if "decoder" in self.fixed_weights:
                self.decoder.fix_variables()
            self.decoder.layout[-1]['filters'] = 3

        if self.math_input_network is None:
            self.math_input_network = cfg.build_math_input(scope="math_input_network")

            if "math" in self.fixed_weights:
                self.math_input_network.fix_variables()

        if self.math_network is None:
            self.math_network = cfg.build_math_network(scope="math_network")

            if "math" in self.fixed_weights:
                self.math_network.fix_variables()

        self._tensors = dict(
            inp=inp,
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            background=background,
            batch_size=tf.shape(inp)[0],
        )
        recorded_tensors = dict(
            batch_size=tf.to_float(self.batch_size),
            float_is_training=self.float_is_training
        )

        losses = dict()

        self._process_labels(labels)

        # --- encode ---

        code = self.encoder(inp, (self.H, self.W, attr_dim), is_training)

        if self.variational:
            code_mean, code_log_std = tf.split(code, 2, axis=-1)
            code_std = tf.exp(code_log_std)
            code, code_kl = yolo_air.normal_vae(code_mean, code_std, self.code_prior_mean, self.code_prior_std)

            self._tensors["code_mean"] = code_mean
            self._tensors["code_std"] = code_std
            self._tensors["code_kl"] = code_kl

            if self.train_kl:
                losses['code_kl'] = tf_mean_sum(self._tensors["code_kl"])

        self._tensors["code"] = code

        # --- decode ---

        reconstruction = self.decoder(code, inp.shape[1:], is_training)
        reconstruction = tf.nn.sigmoid(tf.clip_by_value(reconstruction, -10, 10))
        self._tensors["output"] = reconstruction

        if self.train_reconstruction:
            loss_key = 'xent' if self.xent_loss else 'squared'
            self._tensors['per_pixel_reconstruction_loss'] = yolo_rl.loss_builders[loss_key](reconstruction, inp)
            losses['reconstruction'] = tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])

        # --- predict ---

        _code = code_mean if self.variational else code
        _code = tf.reshape(_code, (self.batch_size * self.HW, self.A))
        math_code = self.math_input_network(_code, self.A, self.is_training)
        self._tensors["math_code"] = tf.reshape(math_code, (self.batch_size, self.H, self.W, self.A))
        math_code = tf.reshape(math_code, (self.batch_size, self.H, self.W, 1, self.A))

        math_code = self.build_math_representation(math_code)

        logits = self.math_network(math_code, self.n_classes, self.is_training)

        if self.math_weight is not None:
            recorded_tensors["raw_loss_math"] = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=self._tensors["targets"],
                    logits=logits
                )
            )
            losses["math"] = self.math_weight * recorded_tensors["raw_loss_math"]

        self._tensors["prediction"] = tf.nn.softmax(logits)

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

        return {
            "tensors": self._tensors,
            "recorded_tensors": recorded_tensors,
            "losses": losses
        }


class SimpleMath_RenderHook(object):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        fetched = self._fetch(updater)

        self._plot_reconstruction(updater, fetched)

    def _fetch(self, updater):
        feed_dict = updater.data_manager.do_val()

        network = updater.network

        to_fetch = dict()

        to_fetch["images"] = network._tensors["inp"]
        to_fetch["output"] = network._tensors["output"]

        if 'prediction' in network._tensors:
            to_fetch["prediction"] = network._tensors["prediction"]
            to_fetch["targets"] = network._tensors["targets"]

        to_fetch = {k: v[:self.N] for k, v in to_fetch.items()}

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)

        return fetched

    def _plot_reconstruction(self, updater, fetched):
        images = fetched['images']
        output = fetched['output']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        sqrt_N = int(np.ceil(np.sqrt(self.N)))

        fig, axes = plt.subplots(sqrt_N, 2*sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(sqrt_N, 2*sqrt_N)
        for n, (pred, gt) in enumerate(zip(output, images)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax = axes[i, 2*j]
            ax.imshow(gt, vmin=0.0, vmax=1.0)

            _target = targets[n]
            _prediction = prediction[n]
            ax.set_title("target={}, prediction={}".format(np.argmax(_target), np.argmax(_prediction)))

            ax = axes[i, 2*j+1]
            ax.imshow(pred, vmin=0.0, vmax=1.0)

        plt.subplots_adjust(left=0, right=1, top=.9, bottom=0, wspace=0.1, hspace=0.2)

        local_step = np.inf if cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
        path = updater.exp_dir.path_for(
            'plots',
            'sampled_reconstruction',
            'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))
        fig.savefig(path)
        plt.close(fig)

        shutil.copyfile(
            path,
            os.path.join(os.path.dirname(path), 'latest_stage{:0>4}.pdf'.format(updater.stage_idx)))


env_config = Config(
    log_name="yolo_math",
    build_env=Env,

    image_shape=(48, 48),
    patch_shape=(14, 14),

    min_digits=1,
    max_digits=11,

    largest_digit=99,
    one_hot=True,
    reductions="sum",
    # reductions="A:sum,M:prod,N:min,X:max,C:len",
)

alg_config = Config(
    get_updater=yolo_rl.YoloRL_Updater,
    build_network=YoloAir_MathNetwork,
    stopping_criteria="math_accuracy,max",
    threshold=1.0,

    build_math_network=SequentialRegressionNetwork,
    build_math_cell=lambda scope: tf.contrib.rnn.LSTMBlockCell(128),
    build_math_output=lambda scope: MLP([100, 100], scope=scope),
    build_math_input=lambda scope: MLP([100, 100], scope=scope),

    math_weight=5.0,
    train_kl=True,
    train_reconstruction=True,
    postprocessing="",

    curriculum=[
        dict(),
    ],
)

config = alg_config.copy()
config.update(env_config)

simple_config = config.copy(
    build_network=SimpleMathNetwork,
    render_hook=SimpleMath_RenderHook(),
    stopping_criteria="math_accuracy,max",
    threshold=1.0,
    build_math_encoder=yolo_rl.Backbone,
    build_math_decoder=yolo_rl.InverseBackbone,
    variational=False,
)
