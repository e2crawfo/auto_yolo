import tensorflow as tf
import numpy as np

from dps import cfg
from dps.utils import Param, prime_factors
from dps.utils.tf import (
    ConvNet, ScopedFunction, MLP, apply_mask_and_group_at_front, tf_shape, apply_object_wise)


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

        layout = [
            dict(filters=self.n_channels, kernel_size=4, strides=(_sh, _sw), padding="RIGHT_ONLY")
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
        # Assumes objects range over all but the first and last dimensions
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


class AttentionLayer(ScopedFunction):
    key_dim = Param()
    value_dim = Param()
    n_heads = Param()
    p_dropout = Param()
    build_mlp = Param()
    build_object_wise = Param()
    n_hidden = Param()

    is_built = False

    K = None
    V = None

    def __init__(self, do_object_wise=True, memory=None, scope=None, **kwargs):
        self.do_object_wise = do_object_wise
        self.memory = memory

        super().__init__(scope=scope)

    def _call(self, signal, is_training, memory=None):
        if not self.is_built:
            self.query_funcs = [self.build_mlp(scope="query_head_{}".format(j)) for j in range(self.n_heads)]
            self.key_funcs = [self.build_mlp(scope="key_head_{}".format(j)) for j in range(self.n_heads)]
            self.value_funcs = [self.build_mlp(scope="value_head_{}".format(j)) for j in range(self.n_heads)]
            self.after_func = self.build_mlp(scope="after")

            if self.do_object_wise:
                self.object_wise_func = self.build_object_wise(scope="object_wise")

            if self.memory is not None:
                self.K = [
                    apply_object_wise(self.key_funcs[j], memory, output_size=self.key_dim, is_training=is_training)
                    for j in range(self.n_heads)]
                self.V = [
                    apply_object_wise(self.value_funcs[j], memory, output_size=self.value_dim, is_training=is_training)
                    for j in range(self.n_heads)]

            self.is_built = True

        n_signal_dim = len(signal.shape)
        assert n_signal_dim in [2, 3]

        if isinstance(memory, tuple):
            # keys and values passed in directly
            K, V = memory
        elif memory is not None:
            # memory is a value that we apply key_funcs and value_funcs to to obtain keys and values
            K = [apply_object_wise(self.key_funcs[j], memory, output_size=self.key_dim, is_training=is_training)
                 for j in range(self.n_heads)]
            V = [apply_object_wise(self.value_funcs[j], memory, output_size=self.value_dim, is_training=is_training)
                 for j in range(self.n_heads)]
        elif self.K is not None:
            K = self.K
            V = self.V
        else:
            # self-attention - `signal` used for queries, keys and values.
            K = [apply_object_wise(self.key_funcs[j], signal, output_size=self.key_dim, is_training=is_training)
                 for j in range(self.n_heads)]
            V = [apply_object_wise(self.value_funcs[j], signal, output_size=self.value_dim, is_training=is_training)
                 for j in range(self.n_heads)]

        head_outputs = []
        for j in range(self.n_heads):
            Q = apply_object_wise(self.query_funcs[j], signal, output_size=self.key_dim, is_training=is_training)

            if n_signal_dim == 2:
                Q = Q[:, None, :]

            attention_logits = tf.matmul(Q, K[j], transpose_b=True) / tf.sqrt(tf.to_float(self.key_dim))
            attention = tf.nn.softmax(attention_logits)
            attended = tf.matmul(attention, V[j])  # (..., n_queries, value_dim)

            if n_signal_dim == 2:
                attended = attended[:, 0, :]

            head_outputs.append(attended)

        head_outputs = tf.concat(head_outputs, axis=-1)

        # `after_func` is applied to the concatenation of the head outputs, and the result is added to the original
        # signal. Next, if `object_wise_func` is not None and `do_object_wise` is True, object_wise_func is
        # applied object wise and in a ResNet-style manner.

        output = apply_object_wise(self.after_func, head_outputs, output_size=self.n_hidden, is_training=is_training)
        output = tf.layers.dropout(output, self.p_dropout, training=is_training)
        signal = tf.contrib.layers.layer_norm(signal + output)

        if self.do_object_wise:
            output = apply_object_wise(self.object_wise_func, signal, output_size=self.n_hidden, is_training=is_training)
            output = tf.layers.dropout(output, self.p_dropout, training=is_training)
            signal = tf.contrib.layers.layer_norm(signal + output)

        return signal


class SpatialAttentionLayer(ScopedFunction):
    """ Now there are no keys and queries.

    For the input we are given data and an array of locations. For the output we are just given
    an array of locations.

    Kind of interesting: this can be viewed as a differentiable way of converting a sparse matrix representation
    of an image to a dense representation of an image, assuming the output locations are the locations of image pixels.
    Input data is a list of locations paired with data, conceptually similar to sparse matrix representations.

    """
    kernel_std = Param()
    p_dropout = Param()
    build_mlp = Param()
    build_object_wise = Param()
    n_hidden = Param()

    is_built = False

    def __init__(self, n_hidden, do_object_wise=True, scope=None, **kwargs):
        self.n_hidden = n_hidden
        self.do_object_wise = do_object_wise

        super().__init__(scope=scope)

    def _call(self, input_signal, input_locs, output_locs, is_training):
        if not self.is_built:
            self.value_func = self.build_mlp(scope="value_func")
            self.after_func = self.build_mlp(scope="after")

            if self.do_object_wise:
                self.object_wise_func = self.build_object_wise(scope="object_wise")

            self.is_built = True

        batch_size, n_inp, _ = tf_shape(input_signal)
        loc_dim = tf_shape(input_locs)[-1]
        n_outp = tf_shape(output_locs)[-2]
        input_locs = tf.broadcast_to(input_locs, (batch_size, n_inp, loc_dim))
        output_locs = tf.broadcast_to(output_locs, (batch_size, n_outp, loc_dim))

        dist = output_locs[:, :, None, :] - input_locs[:, None, :, :]
        proximity = tf.exp(-0.5 * tf.reduce_sum((dist / self.kernel_std)**2, axis=3))
        proximity = proximity / (2 * np.pi)**(0.5 * loc_dim) / self.kernel_std**loc_dim

        V = apply_object_wise(
            self.value_func, input_signal,
            output_size=self.n_hidden, is_training=is_training)  # (batch_size, n_inp, value_dim)

        result = tf.matmul(proximity, V)  # (batch_size, n_outp, value_dim)

        # `after_func` is applied to the concatenation of the head outputs, and the result is added to the original
        # signal. Next, if `object_wise_func` is not None and `do_object_wise` is True, object_wise_func is
        # applied object wise and in a ResNet-style manner.

        output = apply_object_wise(self.after_func, result, output_size=self.n_hidden, is_training=is_training)
        output = tf.layers.dropout(output, self.p_dropout, training=is_training)
        signal = tf.contrib.layers.layer_norm(output)

        if self.do_object_wise:
            output = apply_object_wise(self.object_wise_func, signal, output_size=self.n_hidden, is_training=is_training)
            output = tf.layers.dropout(output, self.p_dropout, training=is_training)
            signal = tf.contrib.layers.layer_norm(signal + output)

        return signal


class SpatialAttentionLayerV2(ScopedFunction):
    """ Now there are no keys and queries.

    For the input we are given data and an array of locations. For the output we are just given
    an array of locations.

    Kind of interesting: this can be viewed as a differentiable way of converting a sparse matrix representation
    of an image to a dense representation of an image, assuming the output locations are the locations of image pixels.
    Input data is a list of locations paired with data, conceptually similar to sparse matrix representations.

    """
    kernel_std = Param()
    build_mlp = Param()
    n_hidden = Param()
    do_object_wise = Param()

    is_built = False

    def _call(self, input_locs, input_features, reference_locs, reference_features, is_training):
        """
        input_features: (B, n_inp, n_hidden)
        input_locs: (B, n_inp, loc_dim)
        reference_locs: (B, n_ref, loc_dim)

        """
        assert (reference_features is not None) == self.do_object_wise

        if not self.is_built:
            self.relation_func = self.build_mlp(scope="relation_func")

            if self.do_object_wise:
                self.object_wise_func = self.build_mlp(scope="object_wise_func")

            self.is_built = True

        loc_dim = tf_shape(input_locs)[-1]
        n_ref = tf_shape(reference_locs)[-2]
        batch_size, n_inp, _ = tf_shape(input_features)

        input_locs = tf.broadcast_to(input_locs, (batch_size, n_inp, loc_dim))
        reference_locs = tf.broadcast_to(reference_locs, (batch_size, n_ref, loc_dim))

        adjusted_locs = input_locs[:, None, :, :] - reference_locs[:, :, None, :]  # (B, n_ref, n_inp, loc_dim)
        adjusted_features = tf.tile(input_features[:, None], (1, n_ref, 1, 1))  # (B, n_ref, n_inp, features_dim)
        relation_input = tf.concat([adjusted_features, adjusted_locs], axis=-1)

        if self.do_object_wise:
            object_wise = apply_object_wise(
                self.object_wise_func, reference_features,
                output_size=self.n_hidden, is_training=is_training)  # (B, n_ref, n_hidden)

            _object_wise = tf.tile(object_wise[:, :, None], (1, 1, n_inp, 1))
            relation_input = tf.concat([relation_input, _object_wise], axis=-1)
        else:
            object_wise = None

        V = apply_object_wise(
            self.relation_func, relation_input,
            output_size=self.n_hidden, is_training=is_training)  # (B, n_ref, n_inp, n_hidden)

        attention_weights = tf.exp(-0.5 * tf.reduce_sum((adjusted_locs / self.kernel_std)**2, axis=3))
        attention_weights = (
            attention_weights / (2 * np.pi) ** (loc_dim / 2) / self.kernel_std**loc_dim)  # (B, n_ref, n_inp)

        result = tf.reduce_sum(V * attention_weights[..., None], axis=2)  # (B, n_ref, n_hidden)

        if self.do_object_wise:
            result += object_wise

        # result = tf.contrib.layers.layer_norm(result)

        return result


class DummySpatialAttentionLayer(SpatialAttentionLayerV2):
    """ A replacement for SpatialAttentionLayerV2 which treats all objects independently. """

    def _call(self, input_locs, input_features, reference_locs, reference_features, is_training):
        """ Assumes input_features and reference_features are identical. """
        assert self.do_object_wise

        if not self.is_built:
            self.object_wise_func = self.build_mlp(scope="object_wise_func")
            self.is_built = True

        object_wise = apply_object_wise(
            self.object_wise_func, reference_features,
            output_size=self.n_hidden, is_training=is_training)  # (B, n_ref, n_hidden)

        return object_wise
