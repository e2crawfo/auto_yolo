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

#define EIGEN_USE_THREADS

#include "render_sprites_ops.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T>
struct RenderSprites2DFunctor<CPUDevice, T>{
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const CPUDevice& d,

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

                   const int n_channels){

    const int sprites_batch_stride = max_sprites * sprite_height * sprite_width * (n_channels + 1);
    const int sprites_sprite_stride = sprite_height * sprite_width * (n_channels + 1);
    const int sprites_row_stride = sprite_width * (n_channels + 1);

    const int scales_batch_stride = 2 * max_sprites;
    const int offsets_batch_stride = 2 * max_sprites;

    const int img_batch_stride = img_height * img_width * n_channels;
    const int img_row_stride = img_width * n_channels;

    const T sprite_height_T = static_cast<T>(sprite_height);
    const T sprite_width_T = static_cast<T>(sprite_width);

    const T img_height_T = static_cast<T>(img_height);
    const T img_width_T = static_cast<T>(img_width);

    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);

    if (max_sprites == 0) {
        memcpy(output, backgrounds, sizeof(T) * batch_size * img_height * img_width * n_channels);
        return;
    }

    auto resample_batches = [&](const int start, const int limit) {

      auto get_sprite_data = [&](const int batch_id,
                                 const int sprite_id,
                                 const int x,
                                 const int y,
                                 const int chan) {

        // Assumes that x and y are in the sprite's co-ordinate system

        const bool point_is_in_range =
            (x >= 0 && y >= 0 && x <= sprite_width - 1 && y <= sprite_height - 1);

        return point_is_in_range
               ? sprites[batch_id * sprites_batch_stride +
                         sprite_id * sprites_sprite_stride +
                         y * sprites_row_stride +
                         x * (n_channels + 1) +
                         chan]
               : zero;
      };

      std::vector<T> weighted_sum(n_channels, zero);

      for (int batch_id = start; batch_id < limit; ++batch_id) {
        for (int img_y = 0; img_y < img_height; ++img_y) {
          const T img_y_T = static_cast<T>(img_y);

          for (int img_x = 0; img_x < img_width; ++img_x) {
            const T img_x_T = static_cast<T>(img_x);

            for(int chan = 0; chan < n_channels; ++chan){
                weighted_sum[chan] = 1.0 * backgrounds[batch_id * img_batch_stride +
                                                      img_y * img_row_stride +
                                                      img_x * n_channels + chan];
            }

            T alpha_sum = 1.0;

            for (int sprite_id = 0; sprite_id < n_sprites[batch_id]; ++sprite_id) {
              const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
              const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];

              const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
              const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];

              // The pixel location represented in the sprites's co-ordinate frame
              const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
              const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

              // The bilinear interpolation function:
              //
              // a) implicitly pads the input data with 0s (hence the unusual checks with {x,y} > -1.0)
              //
              // b) returns 0 when sampling outside the (padded) image.
              //
              // The effect is that the sampled signal smoothly goes to 0 outside the original
              // input domain, rather than presenting a jump discontinuity at the image boundaries.
              //
              const bool within_bounds = x > static_cast<T>(-1.0) &&
                                         y > static_cast<T>(-1.0) &&
                                         x < sprite_width_T &&
                                         y < sprite_height_T;

              if(!within_bounds){
                  continue;
              }

              const int fx = std::floor(static_cast<float>(x));
              const int fy = std::floor(static_cast<float>(y));

              const int cx = fx + 1;
              const int cy = fy + 1;

              const T dx = static_cast<T>(cx) - x;
              const T dy = static_cast<T>(cy) - y;

              const T alpha_fxfy = get_sprite_data(batch_id, sprite_id, fx, fy, n_channels);
              const T alpha_cxcy = get_sprite_data(batch_id, sprite_id, cx, cy, n_channels);
              const T alpha_fxcy = get_sprite_data(batch_id, sprite_id, fx, cy, n_channels);
              const T alpha_cxfy = get_sprite_data(batch_id, sprite_id, cx, fy, n_channels);
              const T alpha = dx * dy * alpha_fxfy +
                              (one - dx) * (one - dy) * alpha_cxcy +
                              dx * (one - dy) * alpha_fxcy +
                              (one - dx) * dy * alpha_cxfy;

              alpha_sum += alpha;

              for (int chan = 0; chan < n_channels; ++chan) {
                const T img_fxfy = get_sprite_data(batch_id, sprite_id, fx, fy, chan);
                const T img_cxcy = get_sprite_data(batch_id, sprite_id, cx, cy, chan);
                const T img_fxcy = get_sprite_data(batch_id, sprite_id, fx, cy, chan);
                const T img_cxfy = get_sprite_data(batch_id, sprite_id, cx, fy, chan);
                const T interp = dx * dy * img_fxfy +
                                 (one - dx) * (one - dy) * img_cxcy +
                                 dx * (one - dy) * img_fxcy +
                                 (one - dx) * dy * img_cxfy;

                weighted_sum[chan] += alpha * interp;
              } // channel
            } // sprite_id

            for(int chan = 0; chan < n_channels; ++chan) {
                output[batch_id * img_batch_stride +
                       img_y * img_row_stride +
                       img_x * n_channels + chan] = weighted_sum[chan] / alpha_sum;
            }

          } // img_x
        } // img_y
      } // batch_id
    };

    // Rough estimate of work for each batch entry.
    // From third_party/tensorflow/core/util/work_sharder.cc we gather that an
    // estimate of the cost of each work unit is needed to correctly shard the
    // workload. Shard assumes each cost unit is 1ns, minimum cost per shard
    // being 10us.
    const int64 cost = max_sprites * img_height * img_width * n_channels * 1000;

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    ::tensorflow::Shard(worker_threads.num_threads,
                        worker_threads.workers,
                        batch_size,
                        cost,
                        resample_batches);
  }
};

}  // namespace functor

template <typename Device, typename T>
class RenderSpritesOp : public ::tensorflow::OpKernel {
 public:
  explicit RenderSpritesOp(::tensorflow::OpKernelConstruction* context) :
      ::tensorflow::OpKernel(context) {}

  void Compute(::tensorflow::OpKernelContext* ctx) override {
    const ::tensorflow::Tensor& sprites = ctx->input(0);
    const ::tensorflow::Tensor& n_sprites = ctx->input(1);
    const ::tensorflow::Tensor& scales = ctx->input(2);
    const ::tensorflow::Tensor& offsets = ctx->input(3);
    const ::tensorflow::Tensor& backgrounds = ctx->input(4);

    const ::tensorflow::TensorShape& sprites_shape = sprites.shape();
    const ::tensorflow::TensorShape& n_sprites_shape = n_sprites.shape();
    const ::tensorflow::TensorShape& scales_shape = scales.shape();
    const ::tensorflow::TensorShape& offsets_shape = offsets.shape();
    const ::tensorflow::TensorShape& backgrounds_shape = backgrounds.shape();

    const int batch_size = sprites_shape.dim_size(0);
    const int max_sprites = sprites_shape.dim_size(1);
    const int sprite_height = sprites_shape.dim_size(2);
    const int sprite_width = sprites_shape.dim_size(3);
    const int sprite_channels = sprites_shape.dim_size(4);

    const int img_height = backgrounds_shape.dim_size(1);
    const int img_width = backgrounds_shape.dim_size(2);
    const int n_channels = backgrounds_shape.dim_size(3);

    // ------ check batch size ------

    OP_REQUIRES(ctx, batch_size == n_sprites_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and n_sprites tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    n_sprites_shape.DebugString()));

    OP_REQUIRES(ctx, batch_size == scales_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and scales tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    scales_shape.DebugString()));

    OP_REQUIRES(ctx, batch_size == offsets_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and offsets tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    offsets_shape.DebugString()));

    OP_REQUIRES(ctx, batch_size == backgrounds_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and backgrounds tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    backgrounds_shape.DebugString()));

    // ------ check n_sprites ------

    OP_REQUIRES(ctx, max_sprites == scales_shape.dim_size(1),
                ::tensorflow::errors::InvalidArgument(
                    "Max sprites (dim 1) of sprites and scales tensor must be the "
                    "same, but input shapes are: ", sprites_shape.DebugString(), ", ",
                    scales_shape.DebugString()));

    OP_REQUIRES(ctx, max_sprites == offsets_shape.dim_size(1),
                ::tensorflow::errors::InvalidArgument(
                    "Max sprites (dim 1) of sprites and offsets tensor must be the "
                    "same, but input shapes are: ", sprites_shape.DebugString(), ", ",
                    offsets_shape.DebugString()));

    // ------ trailing dims ------

    // because sprites have an alpha channel
    OP_REQUIRES(ctx, sprite_channels == n_channels + 1,
                ::tensorflow::errors::InvalidArgument(
                    "Channel dimension for sprites must be one larger than channel "
                    "dimension for backgrounds, but input shapes are: ", sprites_shape.DebugString(),
                    ", ", backgrounds_shape.DebugString()));

    OP_REQUIRES(ctx, scales_shape.dim_size(2) == 2,
                ::tensorflow::errors::InvalidArgument(
                    "Trailing dimension of scales must be 2, "
                    "but input shape is: ", scales_shape.DebugString()));

    OP_REQUIRES(ctx, offsets_shape.dim_size(2) == 2,
                ::tensorflow::errors::InvalidArgument(
                    "Trailing dimension of offsets must be 2, "
                    "but input shape is: ", offsets_shape.DebugString()));

    ::tensorflow::TensorShape output_shape = backgrounds.shape();
    ::tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

    // Execute kernel only for nonempty output; otherwise Eigen crashes on GPU.
    functor::RenderSprites2DFunctor<Device, T>()(ctx,
                                                 ctx->eigen_device<Device>(),

                                                 sprites.flat<T>().data(),
                                                 n_sprites.flat<int>().data(),
                                                 scales.flat<T>().data(),
                                                 offsets.flat<T>().data(),
                                                 backgrounds.flat<T>().data(),

                                                 output->flat<T>().data(),

                                                 batch_size,

                                                 max_sprites,
                                                 sprite_height,
                                                 sprite_width,

                                                 img_height,
                                                 img_width,

                                                 n_channels);

  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RenderSpritesOp);
};


#define REGISTER(TYPE)                       \
  REGISTER_KERNEL_BUILDER(                   \
      Name("RenderSprites")                      \
          .Device(DEVICE_CPU)  \
          .TypeConstraint<TYPE>("T"),        \
      RenderSpritesOp<CPUDevice, TYPE>);

// TF_CALL_half(REGISTER);
// TF_CALL_double(REGISTER);
TF_CALL_float(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA
#define REGISTER(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(Name("RenderSprites")                      \
                              .Device(DEVICE_GPU)  \
                              .TypeConstraint<TYPE>("T"),        \
                          RenderSpritesOp<GPUDevice, TYPE>)
// TF_CALL_double(REGISTER);
TF_CALL_float(REGISTER);
#undef REGISTER
#endif  // GOOGLE_CUDA


namespace functor {

template <typename T>
struct RenderSpritesGrad2DFunctor<CPUDevice, T>{
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const CPUDevice& d,

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

                   const int n_channels){

    // Set gradients to 0, because the kernel incrementally updates the tensor entries by adding partial contributions.
    // (Except for grad_backgrounds, which is only set once)
    memset(grad_sprites, 0, sizeof(T) * batch_size * max_sprites * sprite_height * sprite_width * (n_channels + 1));
    memset(grad_n_sprites, 0, sizeof(T) * batch_size);
    memset(grad_scales, 0, sizeof(T) * batch_size * max_sprites * 2);
    memset(grad_offsets, 0, sizeof(T) * batch_size * max_sprites * 2);

    if (max_sprites == 0){
      memcpy(grad_backgrounds, grad_output, sizeof(T) * batch_size * img_height * img_width * n_channels);
      return;
    }

    int sprites_batch_stride = max_sprites * sprite_height * sprite_width * (n_channels + 1);
    int sprites_sprite_stride = sprite_height * sprite_width * (n_channels + 1);
    int sprites_row_stride = sprite_width * (n_channels + 1);

    int scales_batch_stride = 2 * max_sprites;
    int offsets_batch_stride = 2 * max_sprites;

    int img_batch_stride = img_height * img_width * n_channels;
    int img_row_stride = img_width * n_channels;

    T sprite_height_T = static_cast<T>(sprite_height);
    T sprite_width_T = static_cast<T>(sprite_width);

    T img_height_T = static_cast<T>(img_height);
    T img_width_T = static_cast<T>(img_width);

    T zero = static_cast<T>(0.0);
    T one = static_cast<T>(1.0);

    auto update_grads_for_batches = [&](const int start, const int limit) {

      auto get_sprite_data = [&](const int batch_id,
                                 const int sprite_id,
                                 const int x,
                                 const int y,
                                 const int chan) {

        // Assumes that x and y are in the sprite's co-ordinate system

        const bool point_is_in_range =
            (x >= 0 && y >= 0 && x <= sprite_width - 1 && y <= sprite_height - 1);

        return point_is_in_range
               ? sprites[batch_id * sprites_batch_stride +
                         sprite_id * sprites_sprite_stride +
                         y * sprites_row_stride +
                         x * (n_channels + 1) +
                         chan]
               : zero;
      };

      auto update_grad_sprites = [&](const int batch_id,
                                     const int sprite_id,
                                     const int x,
                                     const int y,
                                     const int chan,
                                     const T value) {

        const bool point_is_in_range =
            (x >= 0 && y >= 0 && x <= sprite_width - 1 && y <= sprite_height - 1);

        if (point_is_in_range){
          grad_sprites[batch_id * sprites_batch_stride +
                       sprite_id * sprites_sprite_stride +
                       y * sprites_row_stride +
                       x * (n_channels + 1) +
                       chan] += value;
        }

      };

      auto update_grad_scales_y = [&](const int batch_id,
                                      const int sprite_id,
                                      const T value) {

        grad_scales[batch_id * scales_batch_stride +
                    sprite_id * 2] += value;
      };

      auto update_grad_scales_x = [&](const int batch_id,
                                      const int sprite_id,
                                      const T value) {

        grad_scales[batch_id * scales_batch_stride +
                    sprite_id * 2 + 1] += value;
      };

      auto update_grad_offsets_y = [&](const int batch_id,
                                       const int sprite_id,
                                       const T value) {

        grad_offsets[batch_id * offsets_batch_stride +
                     sprite_id * 2] += value;
      };

      auto update_grad_offsets_x = [&](const int batch_id,
                                       const int sprite_id,
                                       const T value) {

        grad_offsets[batch_id * offsets_batch_stride +
                     sprite_id * 2 + 1] += value;
      };

      std::vector<T> weighted_sum(n_channels, zero);

      for (int batch_id = start; batch_id < limit; ++batch_id) {

        for (int img_y = 0; img_y < img_height; ++img_y) {
          const T img_y_T = static_cast<T>(img_y);

          for (int img_x = 0; img_x < img_width; ++img_x) {
            const T img_x_T = static_cast<T>(img_x);

            for (int chan = 0; chan < n_channels; ++chan) {
                weighted_sum[chan] = 1.0 * backgrounds[batch_id * img_batch_stride +
                                                      img_y * img_row_stride +
                                                      img_x * n_channels + chan];
            }

            T alpha_sum = 1.0;

            // redo forward pass to compute weighted_sum and alpha_sum
            for (int sprite_id = 0; sprite_id < n_sprites[batch_id]; ++sprite_id) {
              const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
              const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];

              const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
              const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];

              // The pixel location represented in the sprites's co-ordinate frame
              const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
              const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

              const bool within_bounds = x > static_cast<T>(-1.0) &&
                                         y > static_cast<T>(-1.0) &&
                                         x < sprite_width_T &&
                                         y < sprite_height_T;

              if(!within_bounds){
                  continue;
              }

              const int fx = std::floor(static_cast<float>(x));
              const int fy = std::floor(static_cast<float>(y));

              const int cx = fx + 1;
              const int cy = fy + 1;

              const T dx = static_cast<T>(cx) - x;
              const T dy = static_cast<T>(cy) - y;

              const T alpha_fxfy = get_sprite_data(batch_id, sprite_id, fx, fy, n_channels);
              const T alpha_cxcy = get_sprite_data(batch_id, sprite_id, cx, cy, n_channels);
              const T alpha_fxcy = get_sprite_data(batch_id, sprite_id, fx, cy, n_channels);
              const T alpha_cxfy = get_sprite_data(batch_id, sprite_id, cx, fy, n_channels);
              const T alpha = dx * dy * alpha_fxfy +
                              (one - dx) * (one - dy) * alpha_cxcy +
                              dx * (one - dy) * alpha_fxcy +
                              (one - dx) * dy * alpha_cxfy;

              alpha_sum += alpha;

              for (int chan = 0; chan < n_channels; ++chan) {
                const T img_fxfy = get_sprite_data(batch_id, sprite_id, fx, fy, chan);
                const T img_cxcy = get_sprite_data(batch_id, sprite_id, cx, cy, chan);
                const T img_fxcy = get_sprite_data(batch_id, sprite_id, fx, cy, chan);
                const T img_cxfy = get_sprite_data(batch_id, sprite_id, cx, fy, chan);
                const T interp = dx * dy * img_fxfy +
                                 (one - dx) * (one - dy) * img_cxcy +
                                 dx * (one - dy) * img_fxcy +
                                 (one - dx) * dy * img_cxfy;

                weighted_sum[chan] += alpha * interp;
              } // channel
            } // sprite_id - forward pass

            // Now do a second forward pass
            for (int sprite_id = 0; sprite_id < n_sprites[batch_id]; ++sprite_id) {
              const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
              const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];

              const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
              const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];

              // The pixel location represented in the sprites's co-ordinate frame
              const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
              const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

              const bool within_bounds = x > static_cast<T>(-1.0) &&
                                         y > static_cast<T>(-1.0) &&
                                         x < sprite_width_T &&
                                         y < sprite_height_T;

              if(!within_bounds){
                  continue;
              }

              const int fx = std::floor(static_cast<float>(x));
              const int fy = std::floor(static_cast<float>(y));

              const int cx = fx + 1;
              const int cy = fy + 1;

              const T dx = static_cast<T>(cx) - x;
              const T dy = static_cast<T>(cy) - y;

              const T alpha_fxfy = get_sprite_data(batch_id, sprite_id, fx, fy, n_channels);
              const T alpha_cxcy = get_sprite_data(batch_id, sprite_id, cx, cy, n_channels);
              const T alpha_fxcy = get_sprite_data(batch_id, sprite_id, fx, cy, n_channels);
              const T alpha_cxfy = get_sprite_data(batch_id, sprite_id, cx, fy, n_channels);

              const T alpha = dx * dy * alpha_fxfy +
                              (one - dx) * (one - dy) * alpha_cxcy +
                              dx * (one - dy) * alpha_fxcy +
                              (one - dx) * dy * alpha_cxfy;

              const T alpha_y_factor = dx * (alpha_fxcy - alpha_fxfy) + (1 - dx) * (alpha_cxcy - alpha_cxfy);
              const T alpha_x_factor = dy * (alpha_cxfy - alpha_fxfy) + (1 - dy) * (alpha_cxcy - alpha_fxcy);

              const T grad_y_wrt_scale_y = -sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / (scale_y * scale_y);
              const T grad_x_wrt_scale_x = -sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / (scale_x * scale_x);

              const T grad_y_wrt_offset_y = -sprite_height_T / scale_y;
              const T grad_x_wrt_offset_x = -sprite_width_T / scale_x;

              for (int chan = 0; chan < n_channels; ++chan) {
                const T go = grad_output[batch_id * img_batch_stride +
                                         img_y * img_row_stride +
                                         img_x * n_channels + chan];

                const T img_fxfy = get_sprite_data(batch_id, sprite_id, fx, fy, chan);
                const T img_cxcy = get_sprite_data(batch_id, sprite_id, cx, cy, chan);
                const T img_fxcy = get_sprite_data(batch_id, sprite_id, fx, cy, chan);
                const T img_cxfy = get_sprite_data(batch_id, sprite_id, cx, fy, chan);

                const T interp = dx * dy * img_fxfy +
                                 (one - dx) * (one - dy) * img_cxcy +
                                 dx * (one - dy) * img_fxcy +
                                 (one - dx) * dy * img_cxfy;

                // ------ update gradient through alpha ------

                const T alpha_premult = go * (interp / alpha_sum - weighted_sum[chan] / (alpha_sum * alpha_sum));

                update_grad_scales_y(batch_id, sprite_id, alpha_premult * alpha_y_factor * grad_y_wrt_scale_y);
                update_grad_scales_x(batch_id, sprite_id, alpha_premult * alpha_x_factor * grad_x_wrt_scale_x);

                update_grad_offsets_y(batch_id, sprite_id, alpha_premult * alpha_y_factor * grad_y_wrt_offset_y);
                update_grad_offsets_x(batch_id, sprite_id, alpha_premult * alpha_x_factor * grad_x_wrt_offset_x);

                update_grad_sprites(batch_id, sprite_id, fx, fy, n_channels, alpha_premult * dx * dy);
                update_grad_sprites(batch_id, sprite_id, cx, cy, n_channels, alpha_premult * (1-dx) * (1-dy));
                update_grad_sprites(batch_id, sprite_id, fx, cy, n_channels, alpha_premult * dx * (1-dy));
                update_grad_sprites(batch_id, sprite_id, cx, fy, n_channels, alpha_premult * (1-dx) * dy);

                // ------ update gradient through sprites ------

                const T sprite_premult = go * alpha / alpha_sum;

                const T y_factor = dx * (img_fxcy - img_fxfy) + (1 - dx) * (img_cxcy - img_cxfy);
                const T x_factor = dy * (img_cxfy - img_fxfy) + (1 - dy) * (img_cxcy - img_fxcy);

                update_grad_scales_y(batch_id, sprite_id, sprite_premult * y_factor * grad_y_wrt_scale_y);
                update_grad_scales_x(batch_id, sprite_id, sprite_premult * x_factor * grad_x_wrt_scale_x);

                update_grad_offsets_y(batch_id, sprite_id, sprite_premult * y_factor * grad_y_wrt_offset_y);
                update_grad_offsets_x(batch_id, sprite_id, sprite_premult * x_factor * grad_x_wrt_offset_x);

                update_grad_sprites(batch_id, sprite_id, fx, fy, chan, sprite_premult * dx * dy);
                update_grad_sprites(batch_id, sprite_id, cx, cy, chan, sprite_premult * (1-dx) * (1-dy));
                update_grad_sprites(batch_id, sprite_id, fx, cy, chan, sprite_premult * dx * (1-dy));
                update_grad_sprites(batch_id, sprite_id, cx, fy, chan, sprite_premult * (1-dx) * dy);
              } // channel
            } // sprite_id - second pass

            for (int chan = 0; chan < n_channels; ++chan) {
              const T go = grad_output[batch_id * img_batch_stride +
                                       img_y * img_row_stride +
                                       img_x * n_channels + chan];
              grad_backgrounds[batch_id * img_batch_stride +
                               img_y * img_row_stride +
                               img_x * n_channels + chan] = go * 1.0 / alpha_sum;
            }
          } // img_x
        } // img_y
      } // batch_id
    };

    // Rough estimate of work for each batch entry.
    // From third_party/tensorflow/core/util/work_sharder.cc we gather that an
    // estimate of the cost of each work unit is needed to correctly shard the
    // workload. Shard assumes each cost unit is 1ns, minimum cost per shard
    // being 10us.
    const int64 cost = max_sprites * img_height * img_width * n_channels * 1000;

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    ::tensorflow::Shard(worker_threads.num_threads,
                        worker_threads.workers,
                        batch_size,
                        cost,
                        update_grads_for_batches);
  }
};

}  // namespace functor


template <typename Device, typename T>
class RenderSpritesGradOp : public ::tensorflow::OpKernel {
 public:
  explicit RenderSpritesGradOp(::tensorflow::OpKernelConstruction* context) :
      ::tensorflow::OpKernel(context) {}

  void Compute(::tensorflow::OpKernelContext* ctx) override {
    const ::tensorflow::Tensor& sprites = ctx->input(0);
    const ::tensorflow::Tensor& n_sprites = ctx->input(1);
    const ::tensorflow::Tensor& scales = ctx->input(2);
    const ::tensorflow::Tensor& offsets = ctx->input(3);
    const ::tensorflow::Tensor& backgrounds = ctx->input(4);
    const ::tensorflow::Tensor& grad_output = ctx->input(5);

    const ::tensorflow::TensorShape& sprites_shape = sprites.shape();
    const ::tensorflow::TensorShape& n_sprites_shape = n_sprites.shape();
    const ::tensorflow::TensorShape& scales_shape = scales.shape();
    const ::tensorflow::TensorShape& offsets_shape = offsets.shape();
    const ::tensorflow::TensorShape& backgrounds_shape = backgrounds.shape();

    const int batch_size = sprites_shape.dim_size(0);
    const int max_sprites = sprites_shape.dim_size(1);
    const int sprite_height = sprites_shape.dim_size(2);
    const int sprite_width = sprites_shape.dim_size(3);
    const int sprite_channels = sprites_shape.dim_size(4);

    const int img_height = backgrounds_shape.dim_size(1);
    const int img_width = backgrounds_shape.dim_size(2);
    const int n_channels = backgrounds_shape.dim_size(3);

    // ------ check batch size ------

    OP_REQUIRES(ctx, batch_size == n_sprites_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and n_sprites tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    n_sprites_shape.DebugString()));

    OP_REQUIRES(ctx, batch_size == scales_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and scales tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    scales_shape.DebugString()));

    OP_REQUIRES(ctx, batch_size == offsets_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and offsets tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    offsets_shape.DebugString()));

    OP_REQUIRES(ctx, batch_size == backgrounds_shape.dim_size(0),
                ::tensorflow::errors::InvalidArgument(
                    "Batch size of sprites and backgrounds tensor must be the same, "
                    "but input shapes are: ", sprites_shape.DebugString(), ", ",
                    backgrounds_shape.DebugString()));

    // ------ check n_sprites ------

    OP_REQUIRES(ctx, max_sprites == scales_shape.dim_size(1),
                ::tensorflow::errors::InvalidArgument(
                    "Max sprites (dim 1) of sprites and scales tensor must be the "
                    "same, but input shapes are: ", sprites_shape.DebugString(), ", ",
                    scales_shape.DebugString()));

    OP_REQUIRES(ctx, max_sprites == offsets_shape.dim_size(1),
                ::tensorflow::errors::InvalidArgument(
                    "Max sprites (dim 1) of sprites and offsets tensor must be the "
                    "same, but input shapes are: ", sprites_shape.DebugString(), ", ",
                    offsets_shape.DebugString()));

    // ------ trailing dims ------

    // because sprites have an alpha channel
    OP_REQUIRES(ctx, sprite_channels == n_channels + 1,
                ::tensorflow::errors::InvalidArgument(
                    "Trailing dimension of  scales must be 2, "
                    "but input shape is: ", scales_shape.DebugString()));

    OP_REQUIRES(ctx, scales_shape.dim_size(2) == 2,
                ::tensorflow::errors::InvalidArgument(
                    "Trailing dimension of  scales must be 2, "
                    "but input shape is: ", scales_shape.DebugString()));

    OP_REQUIRES(ctx, offsets_shape.dim_size(2) == 2,
                ::tensorflow::errors::InvalidArgument(
                    "Trailing dimension of  offsets must be 2, "
                    "but input shape is: ", offsets_shape.DebugString()));

    const ::tensorflow::TensorShape& grad_output_shape = grad_output.shape();

    OP_REQUIRES(ctx, grad_output_shape == backgrounds_shape,
                ::tensorflow::errors::InvalidArgument(
                   "grad_output shape is not consistent with backgrounds "
                   "shape; it should be ",
                   backgrounds_shape.DebugString(), " but is ",
                   grad_output_shape.DebugString()));

    ::tensorflow::Tensor* grad_sprites = nullptr;
    ::tensorflow::Tensor* grad_n_sprites = nullptr;
    ::tensorflow::Tensor* grad_scales = nullptr;
    ::tensorflow::Tensor* grad_offsets = nullptr;
    ::tensorflow::Tensor* grad_backgrounds = nullptr;

    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, sprites.shape(), &grad_sprites));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, n_sprites.shape(), &grad_n_sprites));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(2, scales.shape(), &grad_scales));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3, offsets.shape(), &grad_offsets));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(4, backgrounds.shape(), &grad_backgrounds));

    functor::RenderSpritesGrad2DFunctor<Device, T>()(ctx,
                                                     ctx->eigen_device<Device>(),

                                                     sprites.flat<T>().data(),
                                                     n_sprites.flat<int>().data(),
                                                     scales.flat<T>().data(),
                                                     offsets.flat<T>().data(),
                                                     backgrounds.flat<T>().data(),
                                                     grad_output.flat<T>().data(),

                                                     grad_sprites->flat<T>().data(),
                                                     grad_n_sprites->flat<T>().data(),
                                                     grad_scales->flat<T>().data(),
                                                     grad_offsets->flat<T>().data(),
                                                     grad_backgrounds->flat<T>().data(),

                                                     batch_size,
                                                     max_sprites,
                                                     sprite_height,
                                                     sprite_width,

                                                     img_height,
                                                     img_width,
                                                     n_channels);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RenderSpritesGradOp);
};

#define REGISTER(TYPE)                       \
  REGISTER_KERNEL_BUILDER(                   \
      Name("RenderSpritesGrad")                  \
          .Device(DEVICE_CPU)  \
          .TypeConstraint<TYPE>("T"),        \
      RenderSpritesGradOp<CPUDevice, TYPE>);

// TF_CALL_half(REGISTER);
// TF_CALL_double(REGISTER);
TF_CALL_float(REGISTER);
#undef REGISTER

#if GOOGLE_CUDA
#define REGISTER(TYPE)                                           \
  REGISTER_KERNEL_BUILDER(Name("RenderSpritesGrad")                  \
                              .Device(DEVICE_GPU)  \
                              .TypeConstraint<TYPE>("T"),        \
                          RenderSpritesGradOp<GPUDevice, TYPE>)
// Disable half and double precision since atomicAdds are not supported
// TF_CALL_half(REGISTER);
// TF_CALL_double(REGISTER);
TF_CALL_float(REGISTER);

#undef REGISTER
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
