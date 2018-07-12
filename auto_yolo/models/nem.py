from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import distributions as dist
from tensorflow.contrib.rnn import RNNCell
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import os
import shutil


from dps import cfg
from dps.utils import Param
from dps.utils.tf import ScopedFunction

from auto_yolo.models.core import AP


# -------------------------------- utils.py -------------------------------------


ACTIVATION_FUNCTIONS = {
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh,
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'linear': lambda x: x,
}


def color_spines(ax, color, lw=2):
    for sn in ['top', 'bottom', 'left', 'right']:
        ax.spines[sn].set_linewidth(lw)
        ax.spines[sn].set_color(color)
        ax.spines[sn].set_visible(True)


def get_gamma_colors(nr_colors):
    hsv_colors = np.ones((nr_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, nr_colors, endpoint=False) + 2/3) % 1.0
    color_conv = hsv_to_rgb(hsv_colors)
    return color_conv


def overview_plot(i, gammas, preds, inputs, corrupted=None, **kwargs):
    T, B, K, H, W, C = gammas.shape
    T -= 1  # the initialization doesn't count as iteration
    corrupted = corrupted if corrupted is not None else inputs
    gamma_colors = get_gamma_colors(K)

    # restrict to sample i and get rid of useless dims
    inputs = inputs[:, i, 0]
    gammas = gammas[:, i, :, :, :, 0]
    if preds.shape[1] != B:
        preds = preds[:, 0]
    preds = preds[:, i]
    corrupted = corrupted[:, i, 0]

    # rescale input range to [0 - 1], assumes input data is [0, 1]
    inputs = np.clip(inputs, 0., 1.)
    preds = np.clip(preds, 0., 1.)
    corrupted = np.clip(corrupted, 0., 1.)

    def plot_img(ax, data, cmap='Greys_r', xlabel=None, ylabel=None, border_color=None):
        if data.shape[-1] == 1:
            ax.matshow(data[:, :, 0], cmap=cmap, vmin=0., vmax=1., interpolation='nearest')
        else:
            ax.imshow(data, interpolation='nearest')

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xlabel(xlabel, color=border_color or 'k') if xlabel else None
        ax.set_ylabel(ylabel, color=border_color or 'k') if ylabel else None
        if border_color:
            color_spines(ax, color=border_color)

    def plot_gamma(ax, gamma, xlabel=None, ylabel=None):
        gamma = np.transpose(gamma, [1, 2, 0])
        gamma = gamma.reshape(-1, gamma.shape[-1]).dot(gamma_colors).reshape(gamma.shape[:-1] + (3,))
        ax.imshow(gamma, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel) if xlabel else None
        ax.set_ylabel(ylabel) if ylabel else None

    # if inputs.shape[0] > 1:
    nrows, ncols = (K + 4, T + 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2 * ncols, 2 * nrows))

    axes[0, 0].set_visible(False)
    axes[1, 0].set_visible(False)
    plot_gamma(axes[2, 0], gammas[0], ylabel='Gammas')
    for k in range(K + 1):
        axes[k + 3, 0].set_visible(False)
    for t in range(1, T + 1):
        g = gammas[t]
        p = preds[t]
        reconst = np.sum(g[:, :, :, None] * p, axis=0)
        plot_img(axes[0, t], inputs[t])
        plot_img(axes[1, t], reconst)
        plot_gamma(axes[2, t], g)
        for k in range(K):
            plot_img(axes[k + 3, t], p[k], border_color=tuple(gamma_colors[k]),
                     ylabel=('mu_{}'.format(k) if t == 1 else None))
        plot_img(axes[K + 3, t], corrupted[t - 1])

    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig


# -------------------------------- network.py ------------------------------------

class _ScopedFunction(ScopedFunction):
    def __init__(self, spec, scope=None):
        self._spec = spec
        super(_ScopedFunction, self).__init__(scope=scope)

    def _call(self, inputs, output_size, is_training):
        inputs = self._subcall(inputs, output_size, is_training)
        if self._spec.get('ln', False):
            inputs = slim.layer_norm(inputs)

        act = self._spec.get('act', False)
        if act:
            activation = ACTIVATION_FUNCTIONS[act]
            return activation(inputs)

        return inputs


class Conv(_ScopedFunction):
    def _subcall(self, inputs, output_size, is_training):
        return slim.conv2d(
            inputs, self._spec['size'], self._spec['kernel'], self._spec['stride'], activation_fn=None)


class TConv(_ScopedFunction):
    def _subcall(self, inputs, output_size, is_training):
        return slim.layers.conv2d_transpose(
            inputs, self._spec['size'], self._spec['kernel'], self._spec['stride'], activation_fn=None)


class RConv(_ScopedFunction):
    def _subcall(self, inputs, output_size, is_training):
        shape = (
            self._spec['stride'][0] * inputs.get_shape()[1].value,
            self._spec['stride'][1] * inputs.get_shape()[2].value
        )
        resized = tf.image.resize_images(inputs, shape, method=1)
        return slim.layers.conv2d(resized, self._spec['size'], self._spec['kernel'], activation_fn=None)


class FC(_ScopedFunction):
    def _subcall(self, inputs, output_size, is_training):
        return slim.fully_connected(inputs, self._spec['size'], activation_fn=None)


class Reshape(_ScopedFunction):
    def _subcall(self, inputs, output_size, is_training):
        shape = self._spec['shape']
        if shape == -1 or shape == "flatten":
            return slim.flatten(inputs)
        else:
            batch_size = tf.shape(inputs)[0]
            return tf.reshape(inputs, (batch_size,) + shape)


class InputNorm(_ScopedFunction):
    def _subcall(self, inputs, output_size, is_training):
        mean, var = tf.nn.moments(inputs, axes=[1])
        inputs = (inputs - tf.expand_dims(mean, axis=1)) / tf.sqrt(tf.expand_dims(var, axis=1))
        return inputs


class FeedforwardNetwork(ScopedFunction):
    functions = dict(
        conv=Conv,
        t_conv=TConv,
        r_conv=RConv,
        fc=FC,
        reshape=Reshape,
        input_norm=InputNorm,
    )

    def __init__(self, layer_specs, scope=None):
        self.layer_specs = layer_specs
        self._layers = None

        super(FeedforwardNetwork, self).__init__(scope=scope)

    def _call(self, inp, output_size, is_training):

        if not self._layers:
            self._layers = []

            for i, spec in enumerate(self.layer_specs):
                name = spec['name']

                layer_name = name + "_" + str(i)

                function = FeedforwardNetwork.functions[name]
                layer = function(spec, scope=layer_name)
                self._layers.append(layer)

        volume = inp
        for layer in self._layers:
            volume = layer(volume, 0, is_training)
        return volume

        # print("**** Handling NEM formulation *****")
        # if use_NEM_formulation:
        #     cell = _NEMCell(recurrent[0]['size'])
        #     cell = tf.contrib.rnn.MultiRNNCell([cell])

        #     cell = NEMOutputWrapper(cell, out_size, "multi_rnn_cell/cell_0/EMCell")
        #     cell = ActivationFunctionWrapper(cell, output[0]['act'])

        #     return cell

        # cell_list = []
        # for i, layer in enumerate(recurrent):
        #     print(sorted([(k, v) for k, v in layer.items()]))
        #     cell = tf.contrib.rnn.BasicRNNCell(layer['size'], activation=ACTIVATION_FUNCTIONS['linear'])
        #     cell = LayerNormWrapper(cell, apply_to='output', name='LayerNormR{}'.format(i)) if layer['ln'] else cell
        #     cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='state')
        #     cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='output')
        #     cell_list.append(cell)
        # cell = tf.contrib.rnn.MultiRNNCell(cell_list)


class FullCell(RNNCell):
    def __init__(self, input_network, cell, output_network, is_training):
        self.input_network = input_network
        self.cell = cell
        self.output_network = output_network
        self.is_training = is_training

    @property
    def state_size(self):
        return self.cell.state_size

    @property
    def output_size(self):
        return self.cell.output_size

    def __call__(self, input, state, scope=None):
        _input = self.input_network(input, 0, self.is_training)
        output, new_state = self.cell(_input, state)
        _output = self.output_network(output, 0, self.is_training)
        return _output, new_state


# EM CELL (WRAPPERS)

class _NEMCell(RNNCell):

    def __init__(self, num_units, name=None):
        self._num_units = num_units
        super(_NEMCell, self).__init__(name=name)

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _call(self, inputs, state, scope=None):
        scope, reuse = self.resolve_scope()

        with tf.variable_scope(scope, reuse=reuse):
            lr = tf.get_variable("scalar", shape=(1, 1), dtype=tf.float32)

            # apply z = z' + lr * sigma(z')(1 - sigma(z'))* W^T * x
            output = state + lr * tf.sigmoid(state) * (1 - tf.sigmoid(state)) * slim.fully_connected(
                inputs, self._num_units, scope='input', activation_fn=None, biases_initializer=None)

        return tf.sigmoid(output), output


class NEMOutputWrapper(RNNCell):
    def __init__(self, cell, size, weight_path, name=None):
        self._size = size
        self._weight_path = weight_path

        super(NEMOutputWrapper, self).__init__(cell, name)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._size

    def _call(self, inputs, state, scope=None):
        output, res_state = self._cell(inputs, state)

        with tf.variable_scope("multi_rnn_cell/cell_0/_NEMCell/input", reuse=True):
            W_t = tf.transpose(tf.get_variable("weights"))

        projected = tf.matmul(output, W_t)

        return projected, res_state


# -------------------------------- nem.py ----------------------------------------


def add_noise(data, noise_prob, noise_type):
    if noise_type in ['None', 'none', None]:
        return data

    shape = tf.stack([s.value if s.value is not None else tf.shape(data)[i]
                     for i, s in enumerate(data.get_shape())])

    if noise_type == 'bitflip':
        noise_dist = dist.Bernoulli(probs=noise_prob, dtype=data.dtype)
        n = noise_dist.sample(shape)
        corrupted = data + n - 2 * data * n  # hacky way of implementing (data XOR n)
    elif noise_type == 'masked_uniform':
        noise_dist = dist.Uniform(low=0., high=1.)
        noise_uniform = noise_dist.sample(shape)

        # sample mask
        mask_dist = dist.Bernoulli(probs=noise_prob, dtype=data.dtype)
        mask = mask_dist.sample(shape)

        # produce output
        corrupted = mask * noise_uniform + (1 - mask) * data
    else:
        raise KeyError('Unknown noise_type "{}"'.format(noise_type))

    corrupted.set_shape(data.get_shape())
    return corrupted


# def set_up_optimizer(loss, optimizer, params, clip_gradients):
#     opt = {
#         'adam': tf.train.AdamOptimizer,
#         'sgd': tf.train.GradientDescentOptimizer,
#         'momentum': tf.train.MomentumOptimizer,
#         'adadelta': tf.train.AdadeltaOptimizer,
#         'adagrad': tf.train.AdagradOptimizer,
#         'rmsprop': tf.train.RMSPropOptimizer
#     }[optimizer](**params)
#
#     # optionally clip gradients by norm
#     grads_and_vars = opt.compute_gradients(loss)
#     if clip_gradients is not None:
#         grads_and_vars = [(tf.clip_by_norm(grad, clip_gradients), var)
#                           for grad, var in grads_and_vars]
#
#     return opt, opt.apply_gradients(grads_and_vars)


# -------------------------------- nem_model.py ---------------------------------


class NEMCell(RNNCell):
    """A RNNCell like implementation of (RN)N-EM."""
    def __init__(self, cell, input_shape, distribution, pred_init, e_sigma, gradient_gamma):
        self.cell = cell
        if not isinstance(input_shape, tf.TensorShape):
            input_shape = tf.TensorShape(input_shape)
        self._input_shape = input_shape
        self.gamma_shape = tf.TensorShape(input_shape.as_list()[:-1] + [1])
        self.distribution = distribution
        self.pred_init = pred_init
        self.e_sigma = e_sigma
        self.gradient_gamma = gradient_gamma

        super(NEMCell, self).__init__()

    @property
    def state_size(self):
        return self.cell.state_size, self._input_shape, self.gamma_shape

    @property
    def output_size(self):
        return self.cell.output_size, self._input_shape, self.gamma_shape

    def init_state(self, batch_size, K, dtype):
        # inner RNN hidden state init
        with tf.name_scope('inner_RNN_init'):
            h = self.cell.zero_state(batch_size * K, dtype)

        # initial prediction (B, K, H, W, C)
        with tf.name_scope('pred_init'):
            pred_shape = tf.stack([batch_size, K] + self._input_shape.as_list())
            pred = tf.ones(shape=pred_shape, dtype=dtype) * self.pred_init

        # initial gamma (B, K, H, W, 1)
        with tf.name_scope('gamma_init'):
            gamma_shape = self.gamma_shape.as_list()
            shape = tf.stack([batch_size, K] + gamma_shape)

            # init with Gaussian distribution
            gamma = tf.abs(tf.random_normal(shape, dtype=dtype))
            gamma /= tf.reduce_sum(gamma, 1, keepdims=True)

            # init with all 1 if K = 1
            if K == 1:
                gamma = tf.ones_like(gamma)

            return h, pred, gamma

    @staticmethod
    def delta_predictions(predictions, data):
        """Compute the derivative of the prediction wrt. to the loss.
        For binary and real with just μ this reduces to (predictions - data).
        :param predictions: (B, K, H, W, C)
           Note: This is a list to later support getting both μ and σ.
        :param data: (B, 1, H, W, C)

        :return: deltas (B, K, H, W, C)
        """
        with tf.name_scope('delta_predictions'):
            return data - predictions  # implicitly broadcasts over K

    @staticmethod
    def mask_rnn_inputs(rnn_inputs, gamma, gradient_gamma):
        """Mask the deltas (inputs to RNN) by gamma.
        :param rnn_inputs: (B, K, H, W, C)
            Note: This is a list to later support multiple inputs
        :param gamma: (B, K, H, W, 1)

        :return: masked deltas (B, K, H, W, C)
        """
        with tf.name_scope('mask_rnn_inputs'):
            if not gradient_gamma:
                gamma = tf.stop_gradient(gamma)

            return rnn_inputs * gamma  # implicitly broadcasts over C

    def run_inner_rnn(self, masked_deltas, h_old):
        with tf.name_scope('reshape_masked_deltas'):
            shape = tf.shape(masked_deltas)
            batch_size = shape[0]
            K = shape[1]
            M = np.prod(self._input_shape.as_list())
            reshaped_masked_deltas = tf.reshape(masked_deltas, tf.stack([batch_size * K, M]))

        preds, h_new = self.cell(reshaped_masked_deltas, h_old)

        return tf.reshape(preds, shape=shape), h_new

    def compute_em_probabilities(self, predictions, data, epsilon=1e-6):
        """Compute pixelwise probability of predictions (wrt. the data).

        :param predictions: (B, K, H, W, C)
        :param data: (B, 1, H, W, C)
        :return: local loss (B, K, H, W, 1)
        """

        with tf.name_scope('em_loss_{}'.format(self.distribution)):
            if self.distribution == 'bernoulli':
                p = predictions
                probs = data * p + (1 - data) * (1 - p)
            elif self.distribution == 'gaussian':
                mu, sigma = predictions, self.e_sigma
                probs = ((1 / tf.sqrt((2 * np.pi * sigma ** 2))) * tf.exp(-(data - mu) ** 2 / (2 * sigma ** 2)))
            else:
                raise ValueError(
                    'Unknown distribution_type: "{}"'.format(self.distribution))

            # sum loss over channels
            probs = tf.reduce_sum(probs, 4, keepdims=True, name='reduce_channels')

            if epsilon > 0:
                # add epsilon to probs in order to prevent 0 gamma
                probs += epsilon

            return probs

    def e_step(self, preds, targets):
        with tf.name_scope('e_step'):
            probs = self.compute_em_probabilities(preds, targets)

            # compute the new gamma (E-step)
            gamma = probs / tf.reduce_sum(probs, 1, keepdims=True)

            return gamma

    def __call__(self, inputs, state, scope=None):
        # unpack
        input_data, target_data = inputs
        h_old, preds_old, gamma_old = state

        # compute difference between prediction and input
        deltas = self.delta_predictions(preds_old, input_data)

        # mask with gamma
        masked_deltas = self.mask_rnn_inputs(deltas, gamma_old, self.gradient_gamma)

        # compute new predictions
        preds, h_new = self.run_inner_rnn(masked_deltas, h_old)

        # compute the new gammas
        gamma = self.e_step(preds, target_data)

        # pack and return
        outputs = (h_new, preds, gamma)
        return outputs, outputs


def compute_prior(distribution, pixel_prior):
    """ Compute the prior over the input data.

    :return: prior (1, 1, 1, 1, 1)
    """

    if distribution == 'bernoulli':
        return tf.constant(pixel_prior['p'], shape=(1, 1, 1, 1, 1), name='prior')
    elif distribution == 'gaussian':
        return tf.constant(pixel_prior['mu'], shape=(1, 1, 1, 1, 1), name='prior')
    else:
        raise KeyError('Unknown distribution: "{}"'.format(distribution))


# log bci
def binomial_cross_entropy_loss(y, t):
    with tf.name_scope('binomial_ce'):
        clipped_y = tf.clip_by_value(y, 1e-6, 1. - 1.e-6)
        return -(t * tf.log(clipped_y) + (1. - t) * tf.log(1. - clipped_y))


# log gaussian
def gaussian_squared_error_loss(mu, sigma, x):
    return (((mu - x)**2) / (2 * tf.clip_by_value(sigma ** 2, 1e-6, 1e6))) + tf.log(tf.clip_by_value(sigma, 1e-6, 1e6))


# compute KL(p1, p2)
def kl_loss_bernoulli(p1, p2):
    with tf.name_scope('KL_loss'):
        return (
            p1 * tf.log(tf.clip_by_value(p1 / tf.clip_by_value(p2, 1e-6, 1e6), 1e-6, 1e6)) +
            (1 - p1) * tf.log(tf.clip_by_value((1-p1)/tf.clip_by_value(1-p2, 1e-6, 1e6), 1e-6, 1e6))
        )


# compute KL(p1, p2)
def kl_loss_gaussian(mu1, mu2, sigma1, sigma2):
    return (
        tf.log(tf.clip_by_value(sigma2/sigma1, 1e-6, 1e6)) +
        (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5
    )


def compute_outer_loss(mu, gamma, target, prior, pixel_distribution, inter_weight, gradient_gamma):
    with tf.name_scope('outer_loss'):
        if pixel_distribution == 'bernoulli':
            intra_loss = binomial_cross_entropy_loss(mu, target)
            inter_loss = kl_loss_bernoulli(prior, mu)
        elif pixel_distribution == 'gaussian':
            intra_loss = gaussian_squared_error_loss(mu, 1.0, target)
            inter_loss = kl_loss_gaussian(mu, prior, 1.0, 1.0)
        else:
            raise KeyError('Unknown pixel_distribution: "{}"'.format(pixel_distribution))

        # weigh losses by gamma and reduce by taking mean across B and sum across H, W, C, K
        # implemented as sum over all then divide by B
        batch_size = tf.to_float(tf.shape(target)[0])

        if gradient_gamma:
            intra_loss = tf.reduce_sum(intra_loss * gamma) / batch_size
            inter_loss = tf.reduce_sum(inter_loss * (1. - gamma)) / batch_size
        else:
            intra_loss = tf.reduce_sum(intra_loss * tf.stop_gradient(gamma)) / batch_size
            inter_loss = tf.reduce_sum(inter_loss * (1. - tf.stop_gradient(gamma))) / batch_size

        total_loss = intra_loss + inter_weight * inter_loss

        return total_loss, intra_loss, inter_loss


def compute_loss_upper_bound(pred, target, pixel_distribution):
    max_pred = tf.reduce_max(pred, axis=1, keepdims=True)
    if pixel_distribution == 'bernoulli':
        loss = binomial_cross_entropy_loss(max_pred, target)
    elif pixel_distribution == 'gaussian':
        loss = gaussian_squared_error_loss(max_pred, 1.0, target)
    else:
        raise KeyError('Unknown pixel_distribution: "{}"'.format(pixel_distribution))

    # reduce losses by taking mean across B and sum across H, W, C, K
    # implemented as sum over all then divide by B
    batch_size = tf.to_float(tf.shape(target)[0])
    loss_upper_bound = tf.reduce_sum(loss) / batch_size

    return loss_upper_bound


def get_loss_step_weights(n_steps, loss_step_weights):
    if loss_step_weights == 'all':
        return [1.0] * n_steps
    elif loss_step_weights == 'last':
        loss_iter_weights = [0.0] * n_steps
        loss_iter_weights[-1] = 1.0
        return loss_iter_weights
    elif isinstance(loss_step_weights, (list, tuple)):
        assert len(loss_step_weights) == n_steps, len(loss_step_weights)
        return loss_step_weights
    else:
        raise KeyError('Unknown loss_iter_weight type: "{}"'.format(loss_step_weights))


class NEM_Network(ScopedFunction):
    binary = Param()
    k = Param()
    n_steps = Param()
    inter_weight = Param()
    gradient_gamma = Param()
    e_sigma = Param()
    pred_init = Param()
    loss_step_weights = Param()
    noise_prob = Param()
    pixel_prior = Param()

    use_NEM_formulation = Param()

    def __init__(self, env, scope=None, **kwargs):
        self.obs_shape = env.datasets['train'].obs_shape
        self.image_height, self.image_width, self.image_depth = self.obs_shape
        # ap_iou_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # self.eval_funcs = {"AP_at_point_{}".format(int(10 * v)): AP(v) for v in ap_iou_values}
        # self.eval_funcs["AP"] = AP(ap_iou_values)
        self.eval_funcs = dict()

        self.input_network = None
        self.cell = None
        self.output_network = None

        super(NEM_Network, self).__init__(scope=scope)

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

    @property
    def float_do_explore(self):
        return self._tensors["float_do_explore"]

    def _call(self, inp, _, is_training):
        inp, labels, background = inp
        return self.build_graph(inp, labels, background, is_training)

    def build_graph(self, inp, labels, background, is_training):
        self._tensors = dict(
            inp=inp,
            annotations=labels[0],
            n_annotations=labels[1],
            is_training=is_training,
            float_is_training=tf.to_float(is_training),
            background=background,
            batch_size=tf.shape(inp)[0]
        )

        self.tiled_inp = tf.tile(
            inp[None, :, None, ...],
            [self.n_steps+1, 1, 1, 1, 1, 1], name="inp")

        noise_type = 'bitflip' if self.binary else 'masked_uniform'
        inp_corrupted = add_noise(self.tiled_inp, self.noise_prob, noise_type)

        # Get dimensions
        input_shape = tf.shape(self.tiled_inp)
        assert input_shape.get_shape()[0].value == 6, (
            "Requires 6D input (T, B, K, H, W, C) but {}".format(input_shape.get_shape()[0].value))

        # T = time, B = batch size, K = number of components, the rest are image size...

        H, W, C = (x.value for x in self.tiled_inp.get_shape()[-3:])

        pixel_dist = 'bernoulli' if self.binary else 'gaussian'

        # inner_cell = build_network(
        #     H * W * C,
        #     output_dist=pixel_dist,
        #     input=self.input_network,
        #     recurrent=self.recurrent_network,
        #     output=self.output_network,
        #     use_NEM_formulation=self.use_NEM_formulation)

        if self.input_network is None:
            self.input_network = cfg.build_input_network(scope="input_network")

        if self.cell is None:
            self.cell = cfg.build_cell(scope="cell")

        if self.output_network is None:
            self.output_network = cfg.build_output_network(scope="output_network")

        inner_cell = FullCell(self.input_network, self.cell, self.output_network, self.is_training)

        nem_cell = NEMCell(
            inner_cell, input_shape=(H, W, C), distribution=pixel_dist,
            pred_init=self.pred_init, e_sigma=self.e_sigma,
            gradient_gamma=self.gradient_gamma)

        prior = compute_prior(pixel_dist, self.pixel_prior)
        hidden_state = nem_cell.init_state(input_shape[1], self.k, dtype=tf.float32)
        outputs = [hidden_state]
        total_losses, upper_bound_losses, intra_losses, inter_losses = [], [], [], []
        loss_step_weights = get_loss_step_weights(self.n_steps, self.loss_step_weights)

        for t, loss_weight in enumerate(loss_step_weights):
            inputs = (inp_corrupted[t], self.tiled_inp[t+1])
            hidden_state, output = nem_cell(inputs, hidden_state)
            theta, pred, gamma = output

            total_loss, intra_loss, inter_loss = compute_outer_loss(
                pred, gamma, self.tiled_inp[t+1], prior, pixel_distribution=pixel_dist,
                inter_weight=self.inter_weight, gradient_gamma=self.gradient_gamma)

            # compute estimated loss upper bound (which doesn't use E-step)
            loss_upper_bound = compute_loss_upper_bound(pred, self.tiled_inp[t+1], pixel_dist)

            total_losses.append(loss_weight * total_loss)

            upper_bound_losses.append(loss_upper_bound)
            intra_losses.append(intra_loss)
            inter_losses.append(inter_loss)
            outputs.append(output)

        thetas, preds, gammas = zip(*outputs)
        self._tensors["thetas"] = tf.stack(thetas)               # (T, 1, B*K, M)
        self._tensors["preds"] = tf.stack(preds)                 # (T, B, K, H, W, C)
        self._tensors["gammas"] = tf.stack(gammas)               # (T, B, K, H, W, C)
        self._tensors["output"] = self._tensors["preds"][-1, :, 0]

        intra_losses = tf.stack(intra_losses)   # (T,)
        inter_losses = tf.stack(inter_losses)   # (T,)
        upper_bound_losses = tf.stack(upper_bound_losses)  # (T,)

        regularization_loss = tf.stack(total_losses)

        flat_inp = tf.layers.flatten(self._tensors["inp"])
        reconstruction = tf.layers.flatten(self._tensors["output"])
        reconstruction_loss = -tf.reduce_sum(
            flat_inp * tf.log(reconstruction + 1e-9) +
            (1.0 - flat_inp) * tf.log(1.0 - reconstruction + 1e-9),
            axis=1, name="reconstruction_loss"
        )

        losses = dict(
            regularization=tf.reduce_mean(regularization_loss),
            reconstruction=tf.reduce_mean(reconstruction_loss)
        )

        self.recorded_tensors = {
            "upper_bound_loss_last": tf.reduce_sum(upper_bound_losses[-1]),
            "intra_loss_last": tf.reduce_sum(intra_losses[-1]),
            "inter_loss_last": tf.reduce_sum(inter_losses[-1]),
        }

        return {
            "tensors": self._tensors,
            "recorded_tensors": self.recorded_tensors,
            "losses": losses,
        }


class NeuralEM_RenderHook(object):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        fetched = self._fetch(self.N, updater)
        self._plot(updater, fetched)

    def _fetch(self, N, updater):
        feed_dict = updater.data_manager.do_val()

        network = updater.network

        to_fetch = dict(
            gammas=network._tensors["gammas"][:, :self.N],
            preds=network._tensors["preds"][:, :self.N],
            images=network._tensors["inp"][:self.N]
        )

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        return fetched

    def _plot(self, updater, fetched):
        images = fetched['images']
        preds = fetched['preds']
        gammas = fetched['gammas']
        hard_gammas = np.argmax(gammas, axis=2)

        N = images.shape[0]
        network = updater.network

        _, image_height, image_width, _ = images.shape

        for i in range(N):
            fig, axes = plt.subplots(2*network.k + 2, network.n_steps+1, figsize=(20, 20))

            for t in range(network.n_steps+1):
                ax = axes[0, t]
                img = images[i]
                ax.imshow(img)
                if t == 0:
                    ax.set_title("ground truth")
                ax.set_xlabel("t = {}".format(t))

                ax = axes[1, t]
                img = preds[t, i, 0]
                ax.imshow(img)
                if t == 0:
                    ax.set_title("reconstruction")
                ax.set_xlabel("t = {}".format(t))

                for k in range(network.k):
                    ax = axes[k+2, t]
                    img = gammas[t, i, k, :, :, 0]
                    ax.imshow(img)
                    if t == 0:
                        ax.set_title("component {} - soft".format(k))
                    ax.set_xlabel("t = {}".format(t))

                for k in range(network.k):
                    ax = axes[network.k + k + 2, t]
                    img = hard_gammas[t, i, :, :, 0] == k
                    ax.imshow(img)
                    if t == 0:
                        ax.set_title("component {} - hard".format(k))
                    ax.set_xlabel("t = {}".format(t))

            local_step = np.inf if cfg.overwrite_plots else "{:0>10}".format(updater.n_updates)
            path = updater.exp_dir.path_for(
                'plots', str(i),
                'stage={:0>4}_local_step={}.pdf'.format(updater.stage_idx, local_step))
            fig.savefig(path)
            plt.close(fig)

            shutil.copyfile(
                path,
                os.path.join(os.path.dirname(path), 'latest_stage{:0>4}.pdf'.format(updater.stage_idx)))
