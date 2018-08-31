import tensorflow as tf
import numpy as np

from dps import cfg
from dps.utils import Param
from dps.utils.tf import (
    ConvNet, ScopedFunction, MLP, apply_mask_and_group_at_front)


class SimpleRecurrentRegressionNetwork(ScopedFunction):
    use_mask = Param()

    cell = None
    output_network = None

    def _call(self, inp, output_size, is_training):
        if self.cell is None:
            self.cell = cfg.build_math_cell(scope="regression_cell")
        if self.output_network is None:
            self.output_network = cfg.build_math_output(scope="math_output")

        if self.use_mask:
            final_dim = int(inp.shape[-1])
            mask, inp = tf.split(inp, (1, final_dim-1), axis=-1)
            inp, n_on, _ = apply_mask_and_group_at_front(inp, mask)
        else:
            batch_size = tf.shape(inp)[0]
            n_objects = np.prod(inp.shape[1:-1])
            A = inp.shape[-1]
            inp = tf.reshape(inp, (batch_size, n_objects, A))

        batch_size = tf.shape(inp)[0]
        output, final_state = tf.nn.dynamic_rnn(
            self.cell, inp, initial_state=self.cell.zero_state(batch_size, tf.float32),
            parallel_iterations=1, swap_memory=False, time_major=False)

        if self.use_mask:
            # Get the output at the end of each sequence.
            indices = tf.stack([tf.range(batch_size), n_on-1], axis=1)
            output = tf.gather_nd(output, indices)
        else:
            output = output[:, -1, :]

        return self.output_network(output, output_size, is_training)


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
