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
import traceback

import tensorflow as tf
from tensorflow.python.framework import ops

load_successful = False
try:
    loc = os.path.join(os.path.split(__file__)[0], "_render_sprites.so")
    print("Loading render_sprites library at {}.".format(loc))
    _render_sprites_so = tf.load_op_library(loc)
except Exception as e:
    print("Failed. Reason:\n{}".format(traceback.format_exc()))
else:
    print("Success.")
    load_successful = True


def lib_avail():
    return load_successful


def render_sprites(sprites, n_sprites, scales, offsets, backgrounds, name="render_sprites"):
  """ Render a scene composed of sprites on top of a background.

  Currently only supports bilinear interpolation.

  An implicit scene is composed by scaling the sprites by `scales`, translating
  them by `offsets`, and putting them on top of the background.

  Args:
    sprites: Tensor of shape `[batch_size, n_sprites, sprite_height, sprite_width, n_channels+1]`
      The final channel is an alpha channel and must be between 0 and 1.
    n_sprites: Tensor of shape `[batch_size,]`
      i-th entry gives number of active sprites for the i-th image (the first i sprites are used)
    scales: Tensor of shape `[batch_size, n_sprites, 2]`
      Amount to scale sprites by. Order is y, x.
    offsets: Tensor of shape `[batch_size, n_sprites, 2]`
      Amount to offset sprites by. Order is y, x.
    backgrounds: Tensor of shape `[batch_size, output_height, output_width, n_channels]`
      The background for each image.
    name: Optional name of the op.

  Returns:
    Tensor giving the stitched images. Shape is
    `[batch_size, output_height, output_width, n_channels]`, same as `backgrounds`.

  Raises:
    ImportError: if the wrapper generated during compilation is not present when
    the function is called.
  """
  with ops.name_scope(name, "render_sprites", [sprites, n_sprites, scales, offsets, backgrounds]):
    sprites_tensor = ops.convert_to_tensor(sprites, name="sprites")
    n_sprites_tensor = ops.convert_to_tensor(n_sprites, dtype=tf.int32, name="n_sprites")
    scales_tensor = ops.convert_to_tensor(scales, name="scales")
    offsets_tensor = ops.convert_to_tensor(offsets, name="offsets")
    backgrounds_tensor = ops.convert_to_tensor(backgrounds, name="backgrounds")
    return _render_sprites_so.render_sprites(
      sprites_tensor, n_sprites_tensor, scales_tensor, offsets_tensor, backgrounds_tensor)


@ops.RegisterGradient("RenderSprites")
def _render_sprites_grad(op, grad_output):
  sprites, n_sprites, scales, offsets, backgrounds = op.inputs
  grad_output_tensor = ops.convert_to_tensor(grad_output, name="grad_output")
  return _render_sprites_so.render_sprites_grad(
    sprites, n_sprites, scales, offsets, backgrounds, grad_output_tensor)


ops.NotDifferentiable("RenderSpritesGrad")
