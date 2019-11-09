import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import gradient_checker
import numpy as np
import pytest
import os
import imageio
import matplotlib as mpl

import dps
from dps.datasets.load import load_backgrounds
from dps.datasets.base import EmnistDataset
from dps.utils import NumpySeed, resize_image

from auto_yolo.tf_ops import render_sprites


def get_session():
    return tf.Session(config=tf.ConfigProto(log_device_placement=True))


def squash_01(val, squash_factor):
    assert ((0 <= val) * (val <= 1)).all()
    val = np.array(val, dtype=np.float32)

    if squash_factor:
        assert squash_factor > 0
        return (val - 0.5) * squash_factor + 0.5
    else:
        return val


def _colorize(img, color=None):
    """ Apply a color to a gray-scale image. """

    color = mpl.colors.to_rgb(color)
    color = np.array(color)[None, None, :]
    color = np.uint8(255. * color)

    rgb = np.tile(color, img.shape + (1,))
    alpha = img[:, :, None]

    return np.concatenate([rgb, alpha], axis=2).astype(np.uint8)


def make_patch(patch_shape, color, shape, importance):
    f = os.path.join(os.path.dirname(dps.__file__), "datasets/shapes", "{}.png".format(shape))
    image = imageio.imread(f)
    image = resize_image(image[..., 3], patch_shape)
    image = _colorize(image, color)
    image = (image / 255.).astype('f')

    imp = np.maximum(importance * image[..., 3:4], 0.01)

    image = np.concatenate([image, imp], axis=2)
    return image


def _get_data():
    image_shape = (100, 100)
    batch_size = 16

    # shapes = ((50, 50), (25, 25), (12, 12), (6, 6))
    # n_shapes = [4, 8, 16, 32]

    # sprite_shapes = [(25, 25)]
    # n_sprites = [2]
    sprite_shapes = [(50, 50), (25, 25), (12, 12)]
    n_sprites = [2, 4, 8]

    n_flights = len(n_sprites)

    shapes = 'circle diamond hollow_circle plus star triangle ud_triangle x'.split()
    colors = list('rgbcmy')
    bg_colors = list('w')

    sprites = [[] for i in range(n_flights)]
    backgrounds = []

    for b in range(batch_size):
        for i, (ss, ns) in enumerate(zip(sprite_shapes, n_sprites)):
            c = np.random.choice(colors, size=ns)
            s = np.random.choice(shapes, size=ns)
            importances = [4**i] * ns

            patches = [make_patch(ss, _c, _s, i) for _c, _s, i in zip(c, s, importances)]
            sprites[i].append(patches)

        bg_color = np.random.choice(bg_colors)
        bg_shape = np.random.choice(shapes)

        bg = make_patch(image_shape, bg_color, bg_shape, 1.0)
        bg = bg[..., :3]

        backgrounds.append(bg)

    sprites = [np.array(s).astype('f') for s in sprites]
    scales = [
        (np.ones((batch_size, ns, 2)) * (np.array(ss) / np.array(image_shape))).astype('f')
        for ss, ns in zip(sprite_shapes, n_sprites)]
    offsets = [0.5 * np.random.rand(batch_size, ns, 2).astype('f') for ns in n_sprites]
    backgrounds = np.array(backgrounds).astype('f')

    return sprites, scales, offsets, backgrounds


def get_data(random_alpha=False, squash=None):
    draw_shape = (56, 56)
    batch_size = 2

    dset = EmnistDataset(classes=[0, 1, 2, 3], include_blank=False, n_examples=100, shape=(28, 28), one_hot=False)

    white = np.array([1., 1., 1.])[None, None, :]
    black = np.array([0., 0., 0.])[None, None, :]
    green = np.array([0., 1., 0.])[None, None, :]
    cyan = np.array([0., 1., 1.])[None, None, :]
    colours = [white, black, green, cyan]
    sprite_pool = [dset.x[list(dset.y).index(idx)][..., None] / 255. for idx in range(4)]
    _sprite_pool = []
    for i, sp in enumerate(sprite_pool):
        colour = colours[i]

        if random_alpha:
            alpha = np.random.rand(*sp[..., :1].shape)
        else:
            alpha = (sp.sum(-1) > 0)[..., None].astype('f')

        alpha = squash_01(alpha, squash)

        sp = colour * sp
        sp = np.concatenate([sp, alpha], axis=-1)
        _sprite_pool.append(sp)

    sprite_pool = _sprite_pool

    first0, first1, first2, first3 = sprite_pool
    sprites0 = np.stack([first0, first1, first2, first3], axis=0)
    sprites1 = np.stack([first3, first2, first1, np.zeros_like(first1)], axis=0)
    sprites = np.stack([sprites0, sprites1], axis=0).astype('f')

    scales = np.ones((batch_size, max_sprites, 2)).astype('f')
    offsets = np.zeros_like(scales)

    backgrounds = np.array(load_backgrounds("red_x blue_circle", draw_shape)) / 255.
    backgrounds = backgrounds.astype('f')

    sprites = squash_01(sprites, squash)
    scales = squash_01(scales, squash)
    offsets = squash_01(offsets, squash)
    backgrounds = squash_01(backgrounds, squash)

    return [sprites], [scales], [offsets], backgrounds


def run(device, show_plots, process_data=None, **get_data_kwargs):
    with NumpySeed(100):
        data = get_data(**get_data_kwargs)

        if process_data is None:
            process_data = lambda *x: x

        sprites, scales, offsets, backgrounds = process_data(*data)

        with tf.device('/{}:0'.format(device)):
            images = render_sprites.render_sprites(sprites, scales, offsets, backgrounds)
            sess = get_session()
            result = sess.run(images)

        result = np.clip(result, 1e-6, 1-1e-6)

    if show_plots:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(result[0])
        ax2.imshow(result[1])
        plt.show()


def visible_gpu():
    d = os.getenv("CUDA_VISIBLE_DEVICES").split(",")[0]
    try:
        d = int(d)
    except Exception:
        return False
    return d >= 0


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
def test_render_sprites_mostly_opaque(device, show_plots):
    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")

    def process_data(sprites, scales, offsets, backgrounds):
        batch_size, max_sprites, *_ = sprites.shape
        sprites[..., 3] = 1.0  # Make the image opaque
        scales = 0.5 * np.ones((batch_size, max_sprites, 2)).astype('f')
        offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
        offsets = np.tile(offsets[None, ...], (batch_size, 1, 1)).astype('f')
        return sprites, scales, offsets, backgrounds

    run(device, show_plots, process_data)


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
def test_render_sprites_background_alpha(device, show_plots):
    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")

    def process_data(sprites, scales, offsets, backgrounds):
        batch_size, max_sprites, *_ = sprites.shape
        scales = 0.5 * np.ones((batch_size, max_sprites, 2)).astype('f')
        offsets = np.array([[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]])
        offsets = np.tile(offsets[None, ...], (batch_size, 1, 1)).astype('f')
        return sprites, scales, offsets, backgrounds

    run(device, show_plots, process_data)


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
def test_render_sprites_overlap(device, show_plots):
    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")
    run(device, show_plots)


@pytest.mark.skipif(not render_sprites.lib_avail(), reason="_render_sprites.so not available")
@pytest.mark.parametrize("device", "cpu gpu".split())
@pytest.mark.slow
def _test_gradient(device):

    if device == "gpu" and visible_gpu():
        pytest.xfail("no gpu is visible")

    with NumpySeed(100):
        with tf.device('/{}:0'.format(device)):
            sprites, scales, offsets, backgrounds = get_data(random_alpha=True, squash=0.99)

            sprites_tf = constant_op.constant(sprites)
            scales_tf = constant_op.constant(scales)
            offsets_tf = constant_op.constant(offsets)
            backgrounds_tf = constant_op.constant(backgrounds)

            images = render_sprites.render_sprites(sprites_tf, scales_tf, offsets_tf, backgrounds_tf)

            sess = get_session()
            with sess.as_default():
                with tf.device(device):
                    err = gradient_checker.compute_gradient_error(
                        [sprites_tf, scales_tf, offsets_tf, backgrounds_tf],
                        [sprites.shape, scales.shape, offsets.shape, backgrounds.shape],
                        images,
                        backgrounds.shape,
                        [sprites, scales, offsets, backgrounds],
                        delta=0.002)

            print("Jacobian error: {}".format(err))
            threshold = 2e-4
            assert err < threshold, "Jacobian error ({}) exceeded threshold ({})".format(err, threshold)


if __name__ == "__main__":

    with NumpySeed(np.random.randint(100000)):
        sprites, scales, offsets, backgrounds = _get_data()

        with tf.device('/{}:0'.format('gpu')):
            images = render_sprites.render_sprites(sprites, scales, offsets, backgrounds)
            sess = get_session()
            result = sess.run(images)

        # Sometimes we get values like 1.0001, nothing really bad.
        result = np.clip(result, 1e-6, 1-1e-6)

        import matplotlib.pyplot as plt
        from dps.utils import square_subplots
        fig, axes = square_subplots(len(sprites[0]))
        for img, ax in zip(result, axes.flatten()):
            ax.imshow(img)
        plt.show()
