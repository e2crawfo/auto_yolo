# pylint: disable=g-bad-file-header
# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tensorflow op performing differentiable resampling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.framework import ops

_render_sprites_so = None


def render_sprites_so():
    global _render_sprites_so
    if _render_sprites_so is None:
        loc = os.path.join(os.path.split(__file__)[0], "_render_sprites.so")
        print("\nLoading render_sprites library at {}.".format(loc))
        _render_sprites_so = tf.load_op_library(loc)
        print("Success.\n")

    return _render_sprites_so


def lib_avail():
    so = render_sprites_so()
    return so is not None


def render_sprites(sprites, scales, offsets, backgrounds, name="render_sprites"):
    """ Render a scene composed of sprites on top of a background.

    An scene is composed by scaling the sprites by `scales` and offseting them by offsets
    (using spatial transformers), and merging the sprites and background together using per-sprite
    alpha and importance channels.

    Sprites are organized into a series of `flights`. Each flight can use a different shape for the sprite maps,
    and there can be a different number of sprites in each flight.

    The coordinate system for scales and offsets has (0, 0) at the image top-left and (1, 1) at the image bottom-right.
    A sprite with scale (1, 1) and offset (0, 0) would occupy the whole output image.

    Uses bilinear interpolation for the spatial transformer sections.

    Args:
      sprites: List of tensors of length `n_flights`, each of shape
        (batch_size, sprite_height_i, sprite_width_i, n_channels+2)
        The sprite maps in flight i are assumed to have shape (sprite_height_i, sprite_width_i).
        The final two channels are the alpha and importance channels.
      scales: Tensor of shape `[batch_size, n_sprites, 2]`
        Amount to scale sprites by. Order is y, x. A value of 1 will have the sprite occupy the whole output image.
      offsets: Tensor of shape `[batch_size, n_sprites, 2]`
        Location of top-left corner of each sprite. Order is y, x.
      backgrounds: Tensor of shape `[batch_size, output_height, output_width, n_channels]`
        The background for each image.
      name: Optional name of the op.

    Returns:
      Tensor giving the stitched images. Shape is
      `(batch_size, output_height, output_width, n_channels)`, same as `backgrounds`.

    Raises:
      ImportError: if the wrapper generated during compilation is not present when
      the function is called.
    """
    with ops.name_scope(name, "render_sprites", [sprites, scales, offsets, backgrounds]):
        sprites_tensor_list = [
            ops.convert_to_tensor(s, name="sprites_flight_{}".format(i))
            for i, s in enumerate(sprites)]
        scales_tensor_list = [
            ops.convert_to_tensor(s, name="scales_flight_{}".format(i))
            for i, s in enumerate(scales)]
        offsets_tensor_list = [
            ops.convert_to_tensor(s, name="offsets_flight_{}".format(i))
            for i, s in enumerate(offsets)]

        backgrounds_tensor = ops.convert_to_tensor(backgrounds, name="backgrounds")

        lib = render_sprites_so()

        output = lib.render_sprites(
            sprites_tensor_list, scales_tensor_list, offsets_tensor_list, backgrounds_tensor)

        return output


@ops.RegisterGradient("RenderSprites")
def _render_sprites_grad(op, grad_output):
    # The grad has to work on a flattened set of tensors; op.inputs is a flattened list of all inputs.
    # In turn, it must return a flattened list of gradients.
    M = len(op.inputs)
    n_flights = (M - 1) // 3
    sprites = op.inputs[:n_flights]
    scales = op.inputs[n_flights:2*n_flights]
    offsets = op.inputs[2*n_flights:3*n_flights]
    backgrounds = op.inputs[3*n_flights]
    grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")
    output = render_sprites_so().render_sprites_grad(
        sprites, scales, offsets, backgrounds, grad_output_tensor, M=M)
    return output


ops.NotDifferentiable("RenderSpritesGrad")
