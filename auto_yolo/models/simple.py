import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from dps import cfg
from dps.utils.tf import tf_mean_sum, RenderHook

from auto_yolo.models.core import loss_builders, normal_vae, VariationalAutoencoder


class SimpleVAE(VariationalAutoencoder):
    encoder = None
    decoder = None

    def build_representation(self):
        # --- init modules ---

        if self.encoder is None:
            self.encoder = cfg.build_encoder(scope="encoder")
            if "encoder" in self.fixed_weights:
                self.encoder.fix_variables()

            if hasattr(self.encoder, "layout"):
                self.encoder.layout[-1]['filters'] = 2 * self.A

        if self.decoder is None:
            self.decoder = cfg.build_decoder(scope="decoder")
            if "decoder" in self.fixed_weights:
                self.decoder.fix_variables()

            if hasattr(self.decoder, "layout"):
                self.decoder.layout[-1]['filters'] = 3

        # --- encode ---

        attr = self.encoder(self.inp, 2 * self.A, self.is_training)
        attr_mean, attr_log_std = tf.split(attr, 2, axis=-1)
        attr_std = tf.exp(attr_log_std)

        if not self.noisy:
            attr_std = tf.zeros_like(attr_std)

        attr, attr_kl = normal_vae(attr_mean, attr_std, self.attr_prior_mean, self.attr_prior_std)

        obj_shape = tf.concat([tf.shape(attr)[:-1], [1]], axis=0)
        self._tensors["obj"] = tf.ones(obj_shape)

        self._tensors.update(attr_mean=attr_mean, attr_std=attr_std, attr_kl=attr_kl, attr=attr)

        # --- decode ---

        reconstruction = self.decoder(attr, self.inp.shape[1:], self.is_training)
        reconstruction = tf.nn.sigmoid(tf.clip_by_value(reconstruction, -10, 10))
        self._tensors["output"] = reconstruction

        # --- losses ---

        if self.train_kl:
            self.losses['attr_kl'] = tf_mean_sum(self._tensors["attr_kl"])

        if self.train_reconstruction:
            loss_key = 'xent' if self.xent_loss else 'squared'
            self._tensors['per_pixel_reconstruction_loss'] = loss_builders[loss_key](reconstruction, self.inp)
            self.losses['reconstruction'] = tf_mean_sum(self._tensors['per_pixel_reconstruction_loss'])


class SimpleVAE_RenderHook(RenderHook):
    def __call__(self, updater):
        self.fetches = "inp output"

        if 'prediction' in updater.network._tensors:
            self.fetches += " prediction targets"

        fetched = self._fetch(updater)
        self._plot_reconstruction(updater, fetched)

    def _plot_reconstruction(self, updater, fetched):
        inp = fetched['inp']
        output = fetched['output']
        prediction = fetched.get("prediction", None)
        targets = fetched.get("targets", None)

        sqrt_N = int(np.ceil(np.sqrt(self.N)))

        fig, axes = plt.subplots(sqrt_N, 2*sqrt_N, figsize=(20, 20))
        axes = np.array(axes).reshape(sqrt_N, 2*sqrt_N)
        for n, (pred, gt) in enumerate(zip(output, inp)):
            i = int(n / sqrt_N)
            j = int(n % sqrt_N)

            ax = axes[i, 2*j]
            self.imshow(ax, gt)

            if targets is not None:
                _target = targets[n]
                _prediction = prediction[n]

                title = "target={}, prediction={}".format(
                    np.argmax(_target), np.argmax(_prediction))
                ax.set_title(title)

            ax = axes[i, 2*j+1]
            self.imshow(ax, pred)

        plt.subplots_adjust(left=0, right=1, top=.9, bottom=0, wspace=0.1, hspace=0.2)
        self.savefig("sampled_reconstruction", fig, updater)
