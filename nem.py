from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib import distributions as dist
from tensorflow.contrib.rnn import RNNCell as _RNNCell
import numpy as np
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import os


from dps import cfg
from dps.datasets import EmnistObjectDetectionDataset
from dps.updater import Updater
from dps.utils import Param, Config, square_subplots
from dps.utils.tf import trainable_variables, build_gradient_train_op


class Env(object):
    def __init__(self):
        train = EmnistObjectDetectionDataset(n_examples=int(cfg.n_train), shuffle=True, example_range=(0.0, 0.9))
        val = EmnistObjectDetectionDataset(n_examples=int(cfg.n_val), shuffle=True, example_range=(0.9, 1.))

        self.datasets = dict(train=train, val=val)

    def close(self):
        pass


# -------------------------------- utils.py ------------------------------------------------


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


# -------------------------------- network.py ------------------------------------------------

class RNNCell(_RNNCell):
    def __call__(self, inputs, state, scope=None):
        output, state = self._call(inputs, state, scope)
        name = getattr(self, "_name", "<None>")

        if isinstance(output, tf.Tensor):
            print("Predicted output size after {} with name {}: {}".format(self.__class__.__name__, name, output.shape))

        if isinstance(state, tf.Tensor):
            print("Predicted state size after {} with name {}: {}".format(self.__class__.__name__, name, state.shape))

        return output, state


class InputWrapper(RNNCell):
    """Adding an input projection to the given cell."""

    def __init__(self, cell, spec, name="InputWrapper"):
        self._cell = cell
        self._spec = spec
        self._name = name

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def _call(self, inputs, state, scope=None):
        projected = None
        with tf.variable_scope(scope or self._name):
            if self._spec['name'] == 'fc':
                projected = slim.fully_connected(inputs, self._spec['size'], activation_fn=None)
            elif self._spec['name'] == 'conv':
                projected = slim.conv2d(inputs, self._spec['size'], self._spec['kernel'], self._spec['stride'], activation_fn=None)
            else:
                raise ValueError('Unknown layer name "{}"'.format(self._spec['name']))

        return self._cell(projected, state)


class OutputWrapper(RNNCell):
    def __init__(self, cell, spec, n_out=1, name="OutputWrapper"):
        self._cell = cell
        self._spec = spec
        self._name = name
        self._n_out = n_out

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._spec['size']

    def _call(self, inputs, state, scope=None):
        output, res_state = self._cell(inputs, state)

        projected = None
        with tf.variable_scope((scope or self._name)):
            if self._spec['name'] == 'fc':
                projected = slim.fully_connected(output, self._spec['size'], activation_fn=None)
            elif self._spec['name'] == 't_conv':
                projected = slim.layers.conv2d_transpose(output, self._spec['size'], self._spec['kernel'], self._spec['stride'], activation_fn=None)
            elif self._spec['name'] == 'r_conv':
                resized = tf.image.resize_images(output, (self._spec['stride'][0] * output.get_shape()[1].value,
                                                          self._spec['stride'][1] * output.get_shape()[2].value), method=1)
                projected = slim.layers.conv2d(resized, self._spec['size'], self._spec['kernel'], activation_fn=None)
            else:
                raise ValueError('Unknown layer name "{}"'.format(self._spec['name']))

        return projected, res_state


class ReshapeWrapper(RNNCell):
    def __init__(self, cell, shape='flatten', apply_to='output'):
        self._cell = cell
        self._shape = shape
        self._apply_to = apply_to

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def _call(self, inputs, state, scope=None):
        batch_size = tf.shape(inputs)[0]

        if self._apply_to == 'input':
            inputs = slim.flatten(inputs) if self._shape == -1 else tf.reshape(inputs, (batch_size,) + self._shape)
            return self._cell(inputs, state)
        elif self._apply_to == 'output':
            output, res_state = self._cell(inputs, state)
            output = slim.flatten(output) if self._shape == -1 else tf.reshape(output, (batch_size,) + self._shape)
            return output, res_state
        elif self._apply_to == 'state':
            output, res_state = self._cell(inputs, state)
            res_state = slim.flatten(res_state) if self._shape == -1 else tf.reshape(res_state, (batch_size,) + self._shape)
            return output, res_state
        else:
            raise ValueError('Unknown apply_to: "{}"'.format(self._apply_to))


class ActivationFunctionWrapper(RNNCell):
    def __init__(self, cell, activation='linear', apply_to='output'):
        self._cell = cell
        self._activation = ACTIVATION_FUNCTIONS[activation]
        self._apply_to = apply_to

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def _call(self, inputs, state, scope=None):
        if self._apply_to == 'input':
            inputs = self._activation(inputs)
            return self._cell(inputs, state)
        elif self._apply_to == 'output':
            output, res_state = self._cell(inputs, state)
            output = self._activation(output)
            return output, res_state
        elif self._apply_to == 'state':
            output, res_state = self._cell(inputs, state)
            res_state = self._activation(res_state)
            return output, res_state
        else:
            raise ValueError('Unknown apply_to: "{}"'.format(self._apply_to))


class LayerNormWrapper(RNNCell):
    def __init__(self, cell, apply_to='output', name="LayerNorm"):
        self._cell = cell
        self._name = name
        self._apply_to = apply_to

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def _call(self, inputs, state, scope=None):
        if self._apply_to == 'input':
            with tf.variable_scope(scope or self._name):
                inputs = slim.layer_norm(inputs)
            return self._cell(inputs, state)
        elif self._apply_to == 'output':
            output, res_state = self._cell(inputs, state)
            with tf.variable_scope(scope or self._name):
                output = slim.layer_norm(output)
                return output, res_state
        elif self._apply_to == 'state':
            output, res_state = self._cell(inputs, state)
            with tf.variable_scope(scope or self._name):
                res_state = slim.layer_norm(res_state)
                return output, res_state
        else:
            raise ValueError('Unknown apply_to: "{}"'.format(self._apply_to))


class InputNormalizationWrapper(RNNCell):
    def __init__(self, cell, name="InputNorm"):
        self._cell = cell
        self._name = name

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def _call(self, inputs, state, scope=None):
        with tf.variable_scope(scope or self._name):
            mean, var = tf.nn.moments(inputs, axes=[1])
            inputs = (inputs - tf.expand_dims(mean, axis=1)) / tf.sqrt(tf.expand_dims(var, axis=1))

        return self._cell(inputs, state)


# EM CELL (WRAPPERS)

class _NEMCell(RNNCell):
    def __init__(self, num_units, name="_NEMCell"):
        self._num_units = num_units
        self._name = name

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def _call(self, inputs, state, scope=None):
        with tf.variable_scope(scope or self._name):
            with tf.variable_scope(scope or "lr"):
                lr = tf.get_variable("scalar", shape=(1, 1), dtype=tf.float32)

            # apply z = z' + lr * sigma(z')(1 - sigma(z'))* W^T * x
            output = state + lr * tf.sigmoid(state) * (1 - tf.sigmoid(state)) * slim.fully_connected(
                inputs, self._num_units, scope='input', activation_fn=None, biases_initializer=None)

        return tf.sigmoid(output), output


class NEMOutputWrapper(RNNCell):
    def __init__(self, cell, size, weight_path, name="NEMOutputWrapper"):
        self._cell = cell
        self._size = size
        self._weight_path = weight_path
        self._name = name

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


# NETWORK BUILDER


def build_network(out_size, output_dist, input, recurrent, output, use_NEM_formulation=False):
    with tf.name_scope('inner_RNN'):
        # use proper mathematical formulation

        print("**** Handling NEM formulation *****")
        if use_NEM_formulation:
            cell = _NEMCell(recurrent[0]['size'])
            cell = tf.contrib.rnn.MultiRNNCell([cell])

            cell = NEMOutputWrapper(cell, out_size, "multi_rnn_cell/cell_0/EMCell")
            cell = ActivationFunctionWrapper(cell, output[0]['act'])

            return cell

        print("**** Handling recurrent *****")

        # build recurrent
        cell_list = []
        for i, layer in enumerate(recurrent):
            print(sorted([(k, v) for k, v in layer.items()]))

            if layer['name'] == 'rnn':
                cell = tf.contrib.rnn.BasicRNNCell(layer['size'], activation=ACTIVATION_FUNCTIONS['linear'])
                cell = LayerNormWrapper(cell, apply_to='output', name='LayerNormR{}'.format(i)) if layer['ln'] else cell
                cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='state')
                cell = ActivationFunctionWrapper(cell, activation=layer['act'], apply_to='output')

            else:
                raise ValueError('Unknown recurrent name "{}"'.format(layer['name']))

            cell_list.append(cell)

        cell = tf.contrib.rnn.MultiRNNCell(cell_list)

        print("**** Handling input *****")

        # build input
        for i, layer in reversed(list(enumerate(input))):
            print(sorted([(k, v) for k, v in layer.items()]))
            if layer['name'] == 'reshape':
                cell = ReshapeWrapper(cell, layer['shape'], apply_to='input')
            elif layer['name'] == 'input_norm':
                cell = InputNormalizationWrapper(cell, name='InputNormalization')
            else:
                cell = ActivationFunctionWrapper(cell, layer['act'], apply_to='input')
                cell = LayerNormWrapper(cell, apply_to='input', name='LayerNormI{}'.format(i)) if layer['ln'] else cell
                cell = InputWrapper(cell, layer, name="InputWrapper{}".format(i))

        print("**** Handling output *****")

        # build output
        for i, layer in enumerate(output):
            print(sorted([(k, v) for k, v in layer.items()]))

            if layer['name'] == 'reshape':
                cell = ReshapeWrapper(cell, layer['shape'])
            else:
                n_out = layer.get('n_out', 1)
                cell = OutputWrapper(cell, layer, n_out=n_out, name="OutputWrapper{}".format(i))
                cell = LayerNormWrapper(cell, apply_to='output', name='LayerNormO{}'.format(i)) if layer['ln'] else cell

                if layer['act'] == '*':
                    output_act = 'linear' if output_dist == 'gaussian' else 'sigmoid'
                    cell = ActivationFunctionWrapper(cell, output_act, apply_to='output')
                else:
                    cell = ActivationFunctionWrapper(cell, layer['act'], apply_to='output')

        return cell


# -------------------------------- nem.py ------------------------------------------------


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


# -------------------------------- nem_model.py ------------------------------------------------


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

    def _call(self, inputs, state, scope=None):
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


class NeuralEM_Updater(Updater):
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
    input_network = Param()
    recurrent_network = Param()
    output_network = Param()

    optimizer_spec = Param()
    lr_schedule = Param()
    noise_schedule = Param()
    max_grad_norm = Param()

    eval_modes = "val".split()

    def __init__(self, env, scope=None, **kwargs):

        self.datasets = env.datasets
        for dset in self.datasets.values():
            dset.reset()

        self.obs_shape = self.datasets['train'].x.shape[1:]
        self.image_height, self.image_width, self.image_depth = self.obs_shape

        self.scope = scope
        self._n_experiences = 0
        self._n_updates = 0

    @property
    def completion(self):
        return self.datasets['train'].completion

    def trainable_variables(self, for_opt, rl_only=False):
        tvars = trainable_variables(self.scope, for_opt=for_opt)
        return tvars

    def _update(self, batch_size, collect_summaries):
        feed_dict = self.make_feed_dict(batch_size, 'train', False)
        sess = tf.get_default_session()

        summary = b''
        if collect_summaries:
            _, record, summary = sess.run(
                [self.train_op, self.recorded_tensors, self.summary_op], feed_dict=feed_dict)
        else:
            _, record = sess.run(
                [self.train_op, self.recorded_tensors], feed_dict=feed_dict)

        return dict(train=(record, summary))

    def _evaluate(self, batch_size, mode):
        assert mode in self.eval_modes

        feed_dict = self.make_feed_dict(None, 'val', True)

        sess = tf.get_default_session()

        record, summary = sess.run(
            [self.recorded_tensors, self.summary_op], feed_dict=feed_dict)

        return record, summary

    def make_feed_dict(self, batch_size, mode, evaluate):
        data = self.datasets[mode].next_batch(batch_size=batch_size, advance=not evaluate)
        if len(data) == 1:
            inp, self.annotations = data[0], None
        elif len(data) == 2:
            inp, self.annotations = data
        else:
            raise Exception()

        return {self.inp_ph: inp, self.is_training: not evaluate}

    def _build_placeholders(self):
        self.inp_ph = tf.placeholder(tf.float32, (None,) + self.obs_shape, name="inp_ph")
        inp = self.inp_ph[None, :, None, ...]
        inp = tf.clip_by_value(inp, 1e-6, 1-1e-6)
        self.inp = tf.tile(inp, [self.n_steps+1, 1, 1, 1, 1, 1], name="inp")

        self.is_training = tf.placeholder(tf.bool, ())
        self.float_is_training = tf.to_float(self.is_training)

        self.batch_size = tf.shape(self.inp)[1]

    def _build_graph(self):
        self._build_placeholders()

        noise_type = 'bitflip' if self.binary else 'masked_uniform'
        inp_corrupted = add_noise(self.inp, self.noise_prob, noise_type)

        # Get dimensions
        input_shape = tf.shape(self.inp)
        assert input_shape.get_shape()[0].value == 6, (
            "Requires 6D input (T, B, K, H, W, C) but {}".format(input_shape.get_shape()[0].value))

        # T = time, B = batch size, K = number of components, the rest are image size...

        H, W, C = (x.value for x in self.inp.get_shape()[-3:])

        # set pixel distribution
        pixel_dist = 'bernoulli' if self.binary else 'gaussian'

        # set up inner cells and nem cells
        inner_cell = build_network(
            H * W * C,
            output_dist=pixel_dist,
            input=self.input_network,
            recurrent=self.recurrent_network,
            output=self.output_network,
            use_NEM_formulation=self.use_NEM_formulation)

        nem_cell = NEMCell(
            inner_cell, input_shape=(H, W, C), distribution=pixel_dist,
            pred_init=self.pred_init, e_sigma=self.e_sigma, gradient_gamma=self.gradient_gamma)

        # compute prior
        prior = compute_prior(pixel_dist, self.pixel_prior)

        # get state initializer
        with tf.name_scope('initial_state'):
            hidden_state = nem_cell.init_state(input_shape[1], self.k, dtype=tf.float32)

        # build static iterations
        outputs = [hidden_state]
        total_losses, upper_bound_losses, intra_losses, inter_losses = [], [], [], []
        loss_step_weights = get_loss_step_weights(self.n_steps, self.loss_step_weights)

        with tf.variable_scope('NEM') as varscope:
            self.var_scope = tf.get_variable_scope()

            for t, loss_weight in enumerate(loss_step_weights):
                varscope.reuse_variables() if t > 0 else None       # share weights across time
                with tf.name_scope('step_{}'.format(t)):
                    # run nem cell
                    inputs = (inp_corrupted[t], self.inp[t+1])
                    hidden_state, output = nem_cell(inputs, hidden_state)
                    theta, pred, gamma = output

                    # compute nem losses
                    total_loss, intra_loss, inter_loss = compute_outer_loss(
                        pred, gamma, self.inp[t+1], prior, pixel_distribution=pixel_dist,
                        inter_weight=self.inter_weight, gradient_gamma=self.gradient_gamma)

                    # compute estimated loss upper bound (which doesn't use E-step)
                    loss_upper_bound = compute_loss_upper_bound(pred, self.inp[t+1], pixel_dist)

                total_losses.append(loss_weight * total_loss)
                upper_bound_losses.append(loss_upper_bound)
                intra_losses.append(intra_loss)
                inter_losses.append(inter_loss)
                outputs.append(output)

        thetas, preds, gammas = zip(*outputs)
        self.thetas = tf.stack(thetas)               # (T, 1, B*K, M)
        self.preds = tf.stack(preds)                 # (T, B, K, H, W, C)
        self.gammas = tf.stack(gammas)               # (T, B, K, H, W, C)

        intra_losses = tf.stack(intra_losses)   # (T,)
        inter_losses = tf.stack(inter_losses)   # (T,)
        upper_bound_losses = tf.stack(upper_bound_losses)  # (T,)

        total_loss = tf.reduce_sum(tf.stack(total_losses))

        tvars = self.trainable_variables(for_opt=True)
        self.train_op, train_summary = build_gradient_train_op(
            total_loss, tvars, self.optimizer_spec, self.lr_schedule,
            self.max_grad_norm, self.noise_schedule)

        self.recorded_tensors = {
            "loss": total_loss,
            "intra_loss_last": tf.reduce_sum(intra_losses[-1]),
            "inter_loss_last": tf.reduce_sum(inter_losses[-1]),
        }

        _summary = [tf.summary.scalar(name, t) for name, t in self.recorded_tensors.items()]
        self.summary_op = tf.summary.merge(_summary + train_summary)

        # create_curve_plots('valid_loss', get_logs('validation.loss'), [0, 2000])


class NeuralEM_RenderHook(object):
    def __init__(self, N=16):
        self.N = N

    def __call__(self, updater):
        if updater.stage_idx == 0:
            path = updater.exp_dir.path_for('plots', 'frames.pdf')
            if not os.path.exists(path):
                fig, axes = square_subplots(16)
                for ax, frame in zip(axes.flatten(), updater.datasets['train'].x):
                    ax.imshow(frame)

                fig.savefig(path)
                plt.close(fig)

        fetched = self._fetch(self.N, updater)
        self._plot(updater, fetched)

    def _fetch(self, N, updater):
        feed_dict = updater.make_feed_dict(N, 'val', True)
        images = feed_dict[updater.inp_ph]

        to_fetch = {"gammas": updater.gammas, "preds": updater.preds}

        sess = tf.get_default_session()
        fetched = sess.run(to_fetch, feed_dict=feed_dict)
        fetched.update(images=images)
        return fetched

    def _plot(self, updater, fetched):
        images = fetched['images']
        preds = fetched['preds']
        gammas = fetched['gammas']
        hard_gammas = np.argmax(gammas, axis=2)

        N = images.shape[0]

        _, image_height, image_width, _ = images.shape

        for i in range(N):
            fig, axes = plt.subplots(2*updater.k + 2, updater.n_steps+1, figsize=(20, 20))

            for t in range(updater.n_steps+1):
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

                for k in range(updater.k):
                    ax = axes[k+2, t]
                    img = gammas[t, i, k, :, :, 0]
                    ax.imshow(img)
                    if t == 0:
                        ax.set_title("component {} - soft".format(k))
                    ax.set_xlabel("t = {}".format(t))

                for k in range(updater.k):
                    ax = axes[updater.k + k + 2, t]
                    img = hard_gammas[t, i, :, :, 0] == k
                    ax.imshow(img)
                    if t == 0:
                        ax.set_title("component {} - hard".format(k))
                    ax.set_xlabel("t = {}".format(t))

            fig.suptitle('Stage={}. After {} experiences ({} updates, {} experiences per batch).'.format(
                updater.stage_idx, updater.n_experiences, updater.n_updates, cfg.batch_size))

            path = updater.exp_dir.path_for('plots', 'stage{}'.format(updater.stage_idx), '{}.pdf'.format(i))
            fig.savefig(path)
            plt.close(fig)


config = Config(
    log_name="nem",
    get_updater=NeuralEM_Updater,
    build_env=Env,

    # env
    min_chars=1,
    max_chars=2,
    characters=[0, 1, 2],
    n_patch_examples=0,
    image_shape=(24, 24),
    patch_shape=(14, 14),
    max_overlap=200,

    optimizer_spec="adam",
    lr_schedule=0.001,
    max_grad_norm=None,

    n_train=1e5,
    n_val=1e2,
    n_test=1e2,

    preserve_env=True,

    eval_step=100,
    display_step=1000,
    batch_size=64,
    max_steps=1e7,

    patience=10000,
    use_gpu=True,
    gpu_allow_growth=True,
    seed=23499123,
    stopping_criteria="loss,min",
    eval_mode="val",

    threshold=-np.inf,
    max_experiments=None,
    render_hook=NeuralEM_RenderHook(4),
    render_step=5000,

    # ------- from nem.py --------

    noise_prob=0.2,                              # probability of annihilating the pixel

    # ------- from nem_model.py ------

    # general
    binary=False,
    gradient_gamma=True,       # whether to back-propagate a gradient through gamma

    # loss
    inter_weight=1.0,          # weight for the inter-cluster loss
    loss_step_weights='last',  # all, last, or list of weights
    pixel_prior=dict(
        p=0.0,                     # probability of success for pixel prior Bernoulli
        mu=0.0,                    # mean of pixel prior Gaussian
        sigma=0.25,                 # std of pixel prior Gaussian
    ),

    # em
    k=3,                       # number of components
    n_steps=10,                # number of (RN)N-EM steps
    e_sigma=0.25,              # sigma used in the e-step when pixel distributions are Gaussian (acts as a temperature)
    pred_init=0.0,             # initial prediction used to compute the input

    # ------- from network.py ------

    use_NEM_formulation=False,

    # input_network=[],
    # recurrent_network=[
    #     {'name': 'rnn', 'size': 250, 'act': 'sigmoid', 'ln': False}
    # ],
    # output_network=[
    #     {'name': 'fc', 'size': 3 * 784, 'act': 'sigmoid', 'ln': False},
    #     # {'name': 'fc', 'size': 784, 'act': '*', 'ln': False},
    # ],

    input_network=[
        {'name': 'input_norm'},
        {'name': 'reshape', 'shape': (24, 24, 3)},
        {'name': 'conv', 'size': 32, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'conv', 'size': 64, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'conv', 'size': 128, 'act': 'elu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'reshape', 'shape': -1},
        {'name': 'fc', 'size': 512, 'act': 'elu', 'ln': True}
    ],
    recurrent_network=[
        {'name': 'rnn', 'size': 250, 'act': 'sigmoid', 'ln': True}
    ],
    output_network=[
        {'name': 'fc', 'size': 512, 'act': 'relu', 'ln': True},
        {'name': 'fc', 'size': 3 * 3 * 128, 'act': 'relu', 'ln': True},
        {'name': 'reshape', 'shape': (3, 3, 128)},
        {'name': 'r_conv', 'size': 64, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'r_conv', 'size': 32, 'act': 'relu', 'stride': [2, 2], 'kernel': (4, 4), 'ln': True},
        {'name': 'r_conv', 'size': 3, 'act': 'sigmoid', 'stride': [2, 2], 'kernel': (4, 4), 'ln': False},
        {'name': 'reshape', 'shape': -1}
    ]
)

reset_config = config.copy(
    load_path="/media/data/dps_data/logs/nem/exp_nem_seed=23499123_2018_04_06_15_11_20/weights/best_of_stage_0",
    do_train=False
)
