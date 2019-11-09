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

#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
class OpKernelContext;
}

namespace tensorflow {
namespace functor {

template<typename T> void allocate_temp_array(
        ::tensorflow::OpKernelContext* ctx, T** ptr_address, int n_elements, Tensor* tensor){

    int n_bytes = n_elements * sizeof(T);
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_UINT8, TensorShape({n_bytes}), tensor));
    *ptr_address = reinterpret_cast<T*>(tensor->flat<unsigned char>().data());
}

template <typename Device, typename T>
struct RenderSprites2DFunctor {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Device& d,

                   const std::vector<int>& shapes,

                   const std::vector<const T*>& sprites,
                   const std::vector<const T*>& scales,
                   const std::vector<const T*>& offsets,
                   const T* __restrict__ backgrounds,

                   T* __restrict__ output,

                   const int batch_size,
                   const int img_height,
                   const int img_width,
                   const int n_channels);
};

template <typename Device, typename T>
struct RenderSpritesGrad2DFunctor {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Device& d,

                   const std::vector<int>& shapes,

                   const std::vector<const T*>& sprites,
                   const std::vector<const T*>& scales,
                   const std::vector<const T*>& offsets,
                   const T* __restrict__ backgrounds,

                   const T* __restrict__ grad_output,

                   const std::vector<T*>& grad_sprites,
                   const std::vector<T*>& grad_scales,
                   const std::vector<T*>& grad_offsets,
                   T* __restrict__ grad_backgrounds,

                   const int batch_size,
                   const int img_height,
                   const int img_width,
                   const int n_channels);
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_RENDER_SPRITES_KERNELS_RENDER_SPRITES_OPS_H_
