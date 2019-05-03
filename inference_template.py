import tensorflow as tf
import numpy as np
import os

from dps import cfg
from dps.config import DEFAULT_CONFIG
from dps.utils import Config
from dps.utils.tf import uninitialized_variables_initializer

import auto_yolo.algs as algs

config = DEFAULT_CONFIG.copy()
config.update(algs.yolo_air_config)
config.update(
    background_cfg=Config(mode="colour", colour="black"),
)


image_shape = (48, 48, 3)
load_path = ""


class Environment:
    @property
    def obs_shape(self):
        return image_shape


env = Environment()


class Updater:
    pass


updater = Updater()

sess = tf.Session()

with config:
    with sess.as_default():
        if hasattr(cfg, 'prepare_func'):
            cfg.prepare_func()

        network = cfg.build_network(env, updater, scope="network")

        inputs = dict(
            image=tf.placeholder(tf.float32, (None, *image_shape)),
        )

        network_outputs = network(inputs, is_training=False)
        network_tensors = network_outputs["tensors"]

        # maybe load weights
        if load_path:
            # variables = {v.name: v for v in trainable_variables("", for_opt=False)}
            # saver = tf.train.Saver(variables)
            saver = tf.train.Saver()
            saver.restore(tf.get_default_session(), os.path.realpath(load_path))

        tf.train.get_or_create_global_step()
        sess.run(uninitialized_variables_initializer())
        sess.run(tf.assert_variables_initialized())

        feed_dict = {inputs["image"]: np.zeros((10, *image_shape))}

        fetches = "obj render_obj z inp output objects n_objects normalized_box glimpse".split()
        to_fetch = {k: network_tensors[k] for k in fetches}

        fetched = sess.run(to_fetch, feed_dict=feed_dict)

        print(fetched)
