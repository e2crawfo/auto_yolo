// Copyright 2017 The Sonnet Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef TENSORFLOW_CONTRIB_RENDER_SPRITES_KERNELS_RENDER_SPRITES_OPS_H_
#define TENSORFLOW_CONTRIB_RENDER_SPRITES_KERNELS_RENDER_SPRITES_OPS_H_

#if PLATFORM_WINDOWS
#define __restrict__ __restrict
#endif

namespace tensorflow {
class OpKernelContext;
}

namespace tensorflow {
namespace functor {


template <typename Device, typename T>
struct RenderSprites2DFunctor {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Device& d,

                   const T* __restrict__ sprites,
                   const int* __restrict__ n_sprites,
                   const T* __restrict__ scales,
                   const T* __restrict__ offsets,
                   const T* __restrict__ backgrounds,

                   T* __restrict__ output,

                   const int batch_size,

                   const int max_sprites,
                   const int sprite_height,
                   const int sprite_width,

                   const int img_height,
                   const int img_width,

                   const int n_channels);
};

template <typename Device, typename T>
struct RenderSpritesGrad2DFunctor {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Device& d,

                   const T* __restrict__ sprites,
                   const int* __restrict__ n_sprites,
                   const T* __restrict__ scales,
                   const T* __restrict__ offsets,
                   const T* __restrict__ backgrounds,
                   const T* __restrict__ grad_output,

                   T* __restrict__ grad_sprites,
                   T* __restrict__ grad_n_sprites,
                   T* __restrict__ grad_scales,
                   T* __restrict__ grad_offsets,
                   T* __restrict__ grad_backgrounds,

                   const int batch_size,

                   const int max_sprites,
                   const int sprite_height,
                   const int sprite_width,

                   const int img_height,
                   const int img_width,

                   const int n_channels);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RENDER_SPRITES_KERNELS_RENDER_SPRITES_OPS_H_
