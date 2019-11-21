import tensorflow as tf

from dps.utils import Parameterized, Param
from dps.utils.tf import tf_shape, tf_binomial_coefficient, build_scheduled_value

from auto_yolo.models.core import concrete_binary_sample_kl, tf_safe_log, logistic_log_pdf


class ObjKL(Parameterized):
    obj_concrete_temp = Param(help="Higher values -> smoother")

    def __init__(self, **kwargs):
        self.obj_concrete_temp = build_scheduled_value(self.obj_concrete_temp, "obj_concrete_temp")
        super().__init__(**kwargs)


class IndependentObjKL(ObjKL):
    prior_log_odds = Param()

    def __init__(self, **kwargs):
        self.prior_log_odds = build_scheduled_value(self.prior_log_odds, "prior_log_odds")
        super().__init__(**kwargs)

    def __call__(self, tensors):
        kl = concrete_binary_sample_kl(
            tensors["obj_pre_sigmoid"],
            tensors["obj_log_odds"], self.obj_concrete_temp,
            self.prior_log_odds, self.obj_concrete_temp)

        batch_size = tf_shape(tensors["obj_pre_sigmoid"])[0]
        return tf.reduce_sum(tf.reshape(kl, (batch_size, -1)), 1)


class SimpleObjKL(ObjKL):
    """ For the prior, we assume that the sum of the concretes are governed by an exponential distribution
        parameterized by `exp_rate` (> 0). As the exponential distribution puts much of its mass near
        0, this encourages the network to use small values for the sum of the concretes.

        A larger value for exp_rate puts more mass near 0, amounting to stronger encouragement for the sum
        of the presence variables to be small.

        Notice that in the end, log_prior_pdf is just the sum of the concretes scaled by exp_rate,
        so exp_rate is sort of a standard regularization weight.

    """
    exp_rate = Param()

    def __init__(self, **kwargs):
        self.exp_rate = build_scheduled_value(self.exp_rate, "exp_rate")
        super().__init__(**kwargs)

    def __call__(self, tensors):
        batch_size = tf_shape(tensors["obj"])[0]

        exp_rate = self.exp_rate
        assert_exp_rate_gt_zero = tf.Assert(exp_rate >= 0, [exp_rate], name='assert_exp_rate_gt_zero')

        with tf.control_dependencies([assert_exp_rate_gt_zero]):
            posterior_log_pdf = logistic_log_pdf(
                tensors["obj_log_odds"], tensors["obj_pre_sigmoid"], self.obj_concrete_temp)
            posterior_log_pdf = tf.reduce_sum(tf.reshape(posterior_log_pdf, (batch_size, -1)), axis=1)

        # This is different from the true log prior pdf by a constant factor,
        # namely the log of the normalization constant for the prior.
        concrete_sum = tf.reduce_sum(tf.reshape(tensors["obj"], (batch_size, -1)), axis=1)

        # prior_pdf = exp_rate * tf.exp(-exp_rate * concrete_sum)
        prior_log_pdf = -exp_rate * concrete_sum

        return posterior_log_pdf - prior_log_pdf


class SequentialObjKL(ObjKL):
    use_concrete_kl = Param()
    count_prior_log_odds = Param()
    count_prior_dist = Param()

    def __init__(self, **kwargs):
        if isinstance(self.count_prior_dist, str):
            self.count_prior_dist = eval(self.count_prior_dist)
        self.count_prior_log_odds = build_scheduled_value(self.count_prior_log_odds, "count_prior_log_odds")
        super().__init__(**kwargs)

    def __call__(self, tensors, existing_objects=None):
        obj_pre_sigmoid = tensors["obj_pre_sigmoid"]
        obj_log_odds = tensors["obj_log_odds"]
        obj_prob = tensors["obj_prob"]
        obj = tensors["obj"]
        batch_size, n_objects, _ = tf_shape(obj)

        max_n_objects = n_objects

        if existing_objects is not None:
            _, n_existing_objects, _ = tf_shape(existing_objects)
            existing_objects = tf.reshape(existing_objects, (batch_size, n_existing_objects))
            max_n_objects += n_existing_objects

        count_support = tf.range(max_n_objects+1, dtype=tf.float32)

        if self.count_prior_dist is not None:
            assert len(self.count_prior_dist) == (max_n_objects + 1)
            count_distribution = tf.constant(self.count_prior_dist, dtype=tf.float32)
        else:
            count_prior_prob = tf.nn.sigmoid(self.count_prior_log_odds)
            count_distribution = count_prior_prob ** count_support

        normalizer = tf.reduce_sum(count_distribution)
        count_distribution = count_distribution / tf.maximum(normalizer, 1e-6)
        count_distribution = tf.tile(count_distribution[None, :], (batch_size, 1))

        if existing_objects is not None:
            count_so_far = tf.reduce_sum(tf.round(existing_objects), axis=1, keepdims=True)

            count_distribution = (
                count_distribution
                * tf_binomial_coefficient(count_support, count_so_far)
                * tf_binomial_coefficient(max_n_objects - count_support, n_existing_objects - count_so_far)
            )

            normalizer = tf.reduce_sum(count_distribution, axis=1, keepdims=True)
            count_distribution = count_distribution / tf.maximum(normalizer, 1e-6)
        else:
            count_so_far = tf.zeros((batch_size, 1), dtype=tf.float32)

        obj_kl = []
        for i in range(n_objects):
            p_z_given_Cz_raw = (count_support[None, :] - count_so_far) / (max_n_objects - i)
            p_z_given_Cz = tf.clip_by_value(p_z_given_Cz_raw, 0.0, 1.0)

            # Doing this instead of 1 - p_z_given_Cz seems to be more numerically stable.
            inv_p_z_given_Cz_raw = (max_n_objects - i - count_support[None, :] + count_so_far) / (max_n_objects - i)
            inv_p_z_given_Cz = tf.clip_by_value(inv_p_z_given_Cz_raw, 0.0, 1.0)

            p_z = tf.reduce_sum(count_distribution * p_z_given_Cz, axis=1, keepdims=True)

            if self.use_concrete_kl:
                prior_log_odds = tf_safe_log(p_z) - tf_safe_log(1-p_z)
                _obj_kl = concrete_binary_sample_kl(
                    obj_pre_sigmoid[:, i, :],
                    obj_log_odds[:, i, :], self.obj_concrete_temp,
                    prior_log_odds, self.obj_concrete_temp,
                )
            else:
                prob = obj_prob[:, i, :]

                _obj_kl = (
                    prob * (tf_safe_log(prob) - tf_safe_log(p_z))
                    + (1-prob) * (tf_safe_log(1-prob) - tf_safe_log(1-p_z))
                )

            obj_kl.append(_obj_kl)

            sample = tf.to_float(obj[:, i, :] > 0.5)
            mult = sample * p_z_given_Cz + (1-sample) * inv_p_z_given_Cz
            raw_count_distribution = mult * count_distribution
            normalizer = tf.reduce_sum(raw_count_distribution, axis=1, keepdims=True)
            normalizer = tf.maximum(normalizer, 1e-6)

            # invalid = tf.logical_and(p_z_given_Cz_raw > 1, count_distribution > 1e-8)
            # float_invalid = tf.cast(invalid, tf.float32)
            # diagnostic = tf.stack(
            #     [float_invalid, p_z_given_Cz, count_distribution, mult, raw_count_distribution], axis=-1)

            # assert_op = tf.Assert(
            #     tf.reduce_all(tf.logical_not(invalid)),
            #     [invalid, diagnostic, count_so_far, sample, tf.constant(i, dtype=tf.float32)],
            #     summarize=100000)

            count_distribution = raw_count_distribution / normalizer
            count_so_far += sample

            # this avoids buildup of inaccuracies that can cause problems in computing p_z_given_Cz_raw
            count_so_far = tf.round(count_so_far)

        obj_kl = tf.reshape(tf.concat(obj_kl, axis=1), (batch_size, n_objects, 1))

        return obj_kl
