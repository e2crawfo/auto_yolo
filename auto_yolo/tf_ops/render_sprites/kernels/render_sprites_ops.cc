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

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
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

                   const std::vector<int>& shapes,

                   const std::vector<const T*>& sprites,
                   const std::vector<const T*>& scales,
                   const std::vector<const T*>& offsets,
                   const T* __restrict__ backgrounds,

                   T* __restrict__ output,

                   const int batch_size,
                   const int img_height,
                   const int img_width,
                   const int n_channels){

    // sprite-level info

    const int n_flights = sprites.size();

    std::vector<int> n_sprites(n_flights);
    std::vector<int> sprite_height(n_flights);
    std::vector<int> sprite_width(n_flights);

    std::vector<T> sprite_height_T(n_flights);
    std::vector<T> sprite_width_T(n_flights);

    std::vector<int> sprites_batch_stride(n_flights);
    std::vector<int> sprites_sprite_stride(n_flights);
    std::vector<int> sprites_row_stride(n_flights);
    std::vector<int> scales_batch_stride(n_flights);

    int n_sprites_per_batch = 0;

    for(int i=0; i < n_flights; i++){
      const int n = shapes[4*i];
      const int h = shapes[4*i+1];
      const int w = shapes[4*i+2];
      const int c = shapes[4*i+3];

      n_sprites_per_batch += n;

      n_sprites[i] = n;
      sprite_height[i] = h;
      sprite_width[i] = w;
      sprite_height_T[i] = static_cast<T>(h);
      sprite_width_T[i] = static_cast<T>(w);

      sprites_batch_stride[i] = n * h * w * c;
      sprites_sprite_stride[i] = h * w * c;
      sprites_row_stride[i] = w * c;

      scales_batch_stride[i] = 2 * n;
    }

    // image-level info

    const int img_batch_stride = img_height * img_width * n_channels;
    const int img_row_stride = img_width * n_channels;

    const T img_height_T = static_cast<T>(img_height);
    const T img_width_T = static_cast<T>(img_width);

    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);

    auto resample_batches = [&](const int start, const int limit) {

      auto get_sprite_data = [&](const int flight_id,
                                 const int batch_id,
                                 const int sprite_id,
                                 const int x,
                                 const int y,
                                 const int chan,
                                 const T default_value=static_cast<T>(0.0)){

        const int h = sprite_height[flight_id];
        const int w = sprite_width[flight_id];

        // Assumes that x and y are in the sprite's co-ordinate system
        const bool point_is_in_range = (x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1);

        return point_is_in_range
               ? sprites[flight_id]
                        [batch_id * sprites_batch_stride[flight_id] +
                         sprite_id * sprites_sprite_stride[flight_id] +
                         y * sprites_row_stride[flight_id] +
                         x * (n_channels + 2) +
                         chan]
               : default_value;
      };

      std::vector<T> weighted_sum(n_channels, zero);
      std::vector<T> bg(n_channels, zero);
      std::vector<T> last_value(n_channels, zero);

      for (int batch_id = start; batch_id < limit; ++batch_id) {

        // --- for each sprite, compute which pixels it affects ---

        std::vector< std::vector<std::pair<int, int>> > affecting(img_width * img_height);

        for (int flight_id = 0; flight_id < n_flights; ++flight_id) {
          const T _left = static_cast<T>(-1.0);
          const T _right = sprite_width_T[flight_id];
          const T _top = static_cast<T>(-1.0);
          const T _bottom = sprite_height_T[flight_id];

          const int sbs = scales_batch_stride[flight_id];

          const T h = sprite_height_T[flight_id];
          const T w = sprite_width_T[flight_id];

          for (int sprite_id = 0; sprite_id < n_sprites[flight_id]; ++sprite_id) {
            const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
            const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];

            const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
            const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];

            const T left = -0.5 + img_width_T * ((_left + 0.5) * scale_x / w + offset_x);
            const T right = -0.5 + img_width_T * ((_right + 0.5) * scale_x / w + offset_x);

            const T top = -0.5 + img_height_T * ((_top + 0.5) * scale_y / h + offset_y);
            const T bottom = -0.5 + img_height_T * ((_bottom + 0.5) * scale_y / h + offset_y);

            const int left_i = static_cast<int>(fmax(0.0, ceil(left)));
            const int right_i = static_cast<int>(fmin(img_width_T-1, floor(right)));

            const int top_i = static_cast<int>(fmax(0.0, ceil(top)));
            const int bottom_i = static_cast<int>(fmin(img_height_T-1, floor(bottom)));

            if (left_i <= right_i && top_i <= bottom_i) {
              for (int i = top_i; i <= bottom_i; i++) {
                for (int j = left_i; j <= right_i; j++) {
                  affecting[i * img_width + j].push_back(std::make_pair(flight_id, sprite_id));
                }
              }
            }
          }
        }

        // --- for each pixel, iterate over all affecting sprites ---

        for (int img_y = 0; img_y < img_height; ++img_y) {
          const T img_y_T = static_cast<T>(img_y);

          for (int img_x = 0; img_x < img_width; ++img_x) {
            const T img_x_T = static_cast<T>(img_x);

            for(int chan = 0; chan < n_channels; ++chan){
                weighted_sum[chan] = 0.0;
                bg[chan] = backgrounds[batch_id * img_batch_stride +
                                       img_y * img_row_stride +
                                       img_x * n_channels + chan];
            }

            T importance_sum = 0.0;
            int n_writes = affecting[img_y * img_width + img_x].size();

            for (auto& idx_pair : affecting[img_y * img_width + img_x]) {
              const int flight_id = idx_pair.first;
              const int sprite_id = idx_pair.second;

              const int sbs = scales_batch_stride[flight_id];

              const T h = sprite_height_T[flight_id];
              const T w = sprite_width_T[flight_id];

              const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
              const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];

              const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
              const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];

              // The pixel location represented in the sprites's co-ordinate frame
              const T y = -0.5 + h * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
              const T x = -0.5 + w * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

              const int fx = std::floor(static_cast<float>(x));
              const int fy = std::floor(static_cast<float>(y));

              const int cx = fx + 1;
              const int cy = fy + 1;

              const T dx = static_cast<T>(cx) - x;
              const T dy = static_cast<T>(cy) - y;

              const T alpha_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels);
              const T alpha_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels);
              const T alpha_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels);
              const T alpha_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels);
              const T alpha = dx * dy * alpha_fxfy +
                              (one - dx) * (one - dy) * alpha_cxcy +
                              dx * (one - dy) * alpha_fxcy +
                              (one - dx) * dy * alpha_cxfy;

              const T imp_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels+1);
              const T imp_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels+1);
              const T imp_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels+1);
              const T imp_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels+1);
              const T imp = dx * dy * imp_fxfy +
                            (one - dx) * (one - dy) * imp_cxcy +
                            dx * (one - dy) * imp_fxcy +
                            (one - dx) * dy * imp_cxfy;

              importance_sum += imp;

              for (int chan = 0; chan < n_channels; ++chan) {
                const T img_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, chan, bg[chan]);
                const T img_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, chan, bg[chan]);
                const T img_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, chan, bg[chan]);
                const T img_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, chan, bg[chan]);
                const T interp = dx * dy * img_fxfy +
                                 (one - dx) * (one - dy) * img_cxcy +
                                 dx * (one - dy) * img_fxcy +
                                 (one - dx) * dy * img_cxfy;

                const T value = alpha * interp + (1-alpha) * bg[chan];
                weighted_sum[chan] += imp * value;
                last_value[chan] = value;
              } // channel
            } // idx_pair

            for(int chan = 0; chan < n_channels; ++chan) {
                if(n_writes == 0){
                    output[batch_id * img_batch_stride +
                           img_y * img_row_stride +
                           img_x * n_channels + chan] = bg[chan];
                }else if(n_writes == 1){
                    output[batch_id * img_batch_stride +
                           img_y * img_row_stride +
                           img_x * n_channels + chan] = last_value[chan];
                }else{
                    output[batch_id * img_batch_stride +
                           img_y * img_row_stride +
                           img_x * n_channels + chan] = weighted_sum[chan] / importance_sum;
                }
            } // channel
          } // img_x
        } // img_y
      } // batch_id
    };

    // Rough estimate of work for each batch entry.
    // From third_party/tensorflow/core/util/work_sharder.cc we gather that an
    // estimate of the cost of each work unit is needed to correctly shard the
    // workload. Shard assumes each cost unit is 1ns, minimum cost per shard
    // being 10us.
    const int64 cost = n_sprites_per_batch * img_height * img_width * n_channels * 1000;

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
  // typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
  //     ConstMatrixVector;

  explicit RenderSpritesOp(::tensorflow::OpKernelConstruction* context) :
      ::tensorflow::OpKernel(context) {}

  void Compute(::tensorflow::OpKernelContext* ctx) override {

    // --- load lists ---

    OpInputList sprites_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("sprites", &sprites_list));

    const int N = sprites_list.size();

    OP_REQUIRES(ctx, N > 0, ::tensorflow::errors::InvalidArgument("List of sprites tensors must be non-empty."));

    OpInputList scales_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("scales", &scales_list));
    OP_REQUIRES(ctx, scales_list.size() == N,
                ::tensorflow::errors::InvalidArgument("scales_list must have same length as sprites_list, but "
                                                      "the two lengths are ", scales_list.size(), " and ", N));

    OpInputList offsets_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("offsets", &offsets_list));
    OP_REQUIRES(ctx, offsets_list.size() == N,
                ::tensorflow::errors::InvalidArgument("offsets_list must have same length as sprites_list, but "
                                                      "the two lengths are ", offsets_list.size(), " and ", N));

    // --- background ---

    const ::tensorflow::Tensor& backgrounds = ctx->input(3*N);
    const ::tensorflow::TensorShape& bg_shape = backgrounds.shape();

    const int batch_size = bg_shape.dim_size(0);
    const int img_height = bg_shape.dim_size(1);
    const int img_width = bg_shape.dim_size(2);
    const int n_channels = bg_shape.dim_size(3);

    // --- store lists and checks dims ---

    std::vector<const T*> sprites, scales, offsets;
    sprites.reserve(N);
    scales.reserve(N);
    offsets.reserve(N);

    std::vector<int> sprite_shapes;
    sprite_shapes.reserve(4*N);

    for (int i = 0; i < N; ++i) {
      const auto& _sprites = sprites_list[i];
      const auto& _scales = scales_list[i];
      const auto& _offsets = offsets_list[i];

      const ::tensorflow::TensorShape& sprites_shape = _sprites.shape();
      const ::tensorflow::TensorShape& scales_shape = _scales.shape();
      const ::tensorflow::TensorShape& offsets_shape = _offsets.shape();

      // --- check n-dims ---

      OP_REQUIRES(ctx, _sprites.dims() == 5,
                  ::tensorflow::errors::InvalidArgument(
                      "sprites tensor for flight ", i, " must have 5 dimensions, "
                      "but shape is: ", sprites_shape.DebugString()));

      OP_REQUIRES(ctx, _scales.dims() == 3,
                  ::tensorflow::errors::InvalidArgument(
                      "scales tensor for flight ", i, " must have 3 dimensions, "
                      "but shape is: ", scales_shape.DebugString()));

      OP_REQUIRES(ctx, _offsets.dims() == 3,
                  ::tensorflow::errors::InvalidArgument(
                      "offsets tensor for flight ", i, " must have 3 dimensions, "
                      "but shape is: ", offsets_shape.DebugString()));

      // ------ check batch size ------
      //
      OP_REQUIRES(ctx, sprites_shape.dim_size(0) == batch_size,
                  ::tensorflow::errors::InvalidArgument(
                      "Batch size for sprites in flight ", i, " must equal batch size for background tensor, "
                      "but shapes are: ", sprites_shape.DebugString(), ", ",
                      bg_shape.DebugString()));

      OP_REQUIRES(ctx, scales_shape.dim_size(0) == batch_size,
                  ::tensorflow::errors::InvalidArgument(
                      "Batch size for scales in flight ", i, " must equal batch size for background tensor, "
                      "but shapes are: ", scales_shape.DebugString(), ", ",
                      bg_shape.DebugString()));

      OP_REQUIRES(ctx, offsets_shape.dim_size(0) == batch_size,
                  ::tensorflow::errors::InvalidArgument(
                      "Batch size for offsets in flight ", i, " must equal batch size for background tensor, "
                      "but shapes are: ", offsets_shape.DebugString(), ", ",
                      bg_shape.DebugString()));

      // ------ check n_sprites ------

      const int n_sprites = sprites_shape.dim_size(1);

      OP_REQUIRES(ctx, scales_shape.dim_size(1) == n_sprites,
                  ::tensorflow::errors::InvalidArgument(
                      "Max sprites (dim 1) of sprites and scales tensor must be the "
                      "same, but shapes for flight ", i, " are: ",
                      sprites_shape.DebugString(), ", ", scales_shape.DebugString()));

      OP_REQUIRES(ctx, offsets_shape.dim_size(1) == n_sprites,
                  ::tensorflow::errors::InvalidArgument(
                      "Max sprites (dim 1) of sprites and offsets tensor must be the "
                      "same, but shapes for flight ", i, " are: ",
                      sprites_shape.DebugString(), ", ", offsets_shape.DebugString()));

      // ------ trailing dims ------

      // because sprites have an alpha and importance channels
      OP_REQUIRES(ctx, sprites_shape.dim_size(4) == n_channels + 2,
                  ::tensorflow::errors::InvalidArgument(
                      "Trailing dimension of sprites must be n_channels + 2, "
                      "but shape for flight ", i, " is : ", sprites_shape.DebugString(),
                      " while n_channels is ", n_channels));

      OP_REQUIRES(ctx, scales_shape.dim_size(2) == 2,
                  ::tensorflow::errors::InvalidArgument(
                      "Trailing dimension of scales must be 2, "
                      "but shape for flight ", i, " is : ", scales_shape.DebugString()));

      OP_REQUIRES(ctx, offsets_shape.dim_size(2) == 2,
                  ::tensorflow::errors::InvalidArgument(
                      "Trailing dimension of offsets must be 2, "
                      "but shape for flight ", i, " is : ", offsets_shape.DebugString()));

      sprites.push_back(_sprites.flat<T>().data());
      scales.push_back(_scales.flat<T>().data());
      offsets.push_back(_offsets.flat<T>().data());

      sprite_shapes.push_back(sprites_shape.dim_size(1));
      sprite_shapes.push_back(sprites_shape.dim_size(2));
      sprite_shapes.push_back(sprites_shape.dim_size(3));
      sprite_shapes.push_back(sprites_shape.dim_size(4));
    }

    ::tensorflow::Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, bg_shape, &output));

    // Execute kernel only for nonempty output; otherwise Eigen crashes on GPU.
    functor::RenderSprites2DFunctor<Device, T>()(ctx,
                                                 ctx->eigen_device<Device>(),

                                                 sprite_shapes,

                                                 sprites,
                                                 scales,
                                                 offsets,

                                                 backgrounds.flat<T>().data(),

                                                 output->flat<T>().data(),

                                                 batch_size,
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
                   const int n_channels){

    // sprite-level info

    const int n_flights = sprites.size();

    std::vector<int> n_sprites(n_flights);
    std::vector<int> sprite_height(n_flights);
    std::vector<int> sprite_width(n_flights);

    std::vector<T> sprite_height_T(n_flights);
    std::vector<T> sprite_width_T(n_flights);

    std::vector<int> sprites_batch_stride(n_flights);
    std::vector<int> sprites_sprite_stride(n_flights);
    std::vector<int> sprites_row_stride(n_flights);
    std::vector<int> scales_batch_stride(n_flights);

    int n_sprites_per_batch = 0;

    for(int i=0; i < n_flights; i++){
      const int n = shapes[4*i];
      const int h = shapes[4*i+1];
      const int w = shapes[4*i+2];
      const int c = shapes[4*i+3];

      n_sprites_per_batch += n;

      n_sprites[i] = n;
      sprite_height[i] = h;
      sprite_width[i] = w;
      sprite_height_T[i] = static_cast<T>(h);
      sprite_width_T[i] = static_cast<T>(w);

      sprites_batch_stride[i] = n * h * w * c;
      sprites_sprite_stride[i] = h * w * c;
      sprites_row_stride[i] = w * c;

      scales_batch_stride[i] = 2 * n;

      // Set gradients to 0, because the kernel incrementally updates the tensor entries by adding partial contributions.
      // (Except for grad_backgrounds, which is only set once)
      memset(grad_sprites[i], 0, sizeof(T) * batch_size * n * h * w * c);
      memset(grad_scales[i], 0, sizeof(T) * batch_size * n * 2);
      memset(grad_offsets[i], 0, sizeof(T) * batch_size * n * 2);
    }

    // image-level info

    const int img_batch_stride = img_height * img_width * n_channels;
    const int img_row_stride = img_width * n_channels;

    const T img_height_T = static_cast<T>(img_height);
    const T img_width_T = static_cast<T>(img_width);

    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);

    auto update_grads_for_batches = [&](const int start, const int limit){

      auto get_sprite_data = [&](const int flight_id,
                                 const int batch_id,
                                 const int sprite_id,
                                 const int x,
                                 const int y,
                                 const int chan,
                                 const T default_value=static_cast<T>(0.0)){

        const int h = sprite_height[flight_id];
        const int w = sprite_width[flight_id];

        // Assumes that x and y are in the sprite's co-ordinate system
        const bool point_is_in_range = (x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1);

        return point_is_in_range
               ? sprites[flight_id]
                        [batch_id * sprites_batch_stride[flight_id] +
                         sprite_id * sprites_sprite_stride[flight_id] +
                         y * sprites_row_stride[flight_id] +
                         x * (n_channels + 2) +
                         chan]
               : default_value;
      };

      auto update_grad_sprites = [&](const int flight_id,
                                     const int batch_id,
                                     const int sprite_id,
                                     const int x,
                                     const int y,
                                     const int chan,
                                     const T value) {

        const int h = sprite_height[flight_id];
        const int w = sprite_width[flight_id];

        // Assumes that x and y are in the sprite's co-ordinate system
        const bool point_is_in_range = (x >= 0 && y >= 0 && x <= w - 1 && y <= h - 1);

        if (point_is_in_range){
          grad_sprites[flight_id]
                      [batch_id * sprites_batch_stride[flight_id] +
                       sprite_id * sprites_sprite_stride[flight_id] +
                       y * sprites_row_stride[flight_id] +
                       x * (n_channels + 2) +
                       chan] += value;
        }

      };

      auto update_grad_scales_y = [&](const int flight_id,
                                      const int batch_id,
                                      const int sprite_id,
                                      const T value) {

        grad_scales[flight_id]
                   [batch_id * scales_batch_stride[flight_id] +
                    sprite_id * 2] += value;
      };

      auto update_grad_scales_x = [&](const int flight_id,
                                      const int batch_id,
                                      const int sprite_id,
                                      const T value) {

        grad_scales[flight_id]
                   [batch_id * scales_batch_stride[flight_id] +
                    sprite_id * 2 + 1] += value;
      };

      auto update_grad_offsets_y = [&](const int flight_id,
                                       const int batch_id,
                                       const int sprite_id,
                                       const T value) {

        grad_offsets[flight_id]
                    [batch_id * scales_batch_stride[flight_id] +
                     sprite_id * 2] += value;
      };

      auto update_grad_offsets_x = [&](const int flight_id,
                                       const int batch_id,
                                       const int sprite_id,
                                       const T value) {

        grad_offsets[flight_id]
                    [batch_id * scales_batch_stride[flight_id] +
                     sprite_id * 2 + 1] += value;
      };

      std::vector<T> weighted_sum(n_channels, zero);
      std::vector<T> bg(n_channels, zero);
      std::vector<T> last_value(n_channels, zero);

      for (int batch_id = start; batch_id < limit; ++batch_id) {

        // --- for each sprite, compute which pixels it affects ---

        std::vector< std::vector<std::pair<int, int>> > affecting(img_width * img_height);

        for (int flight_id = 0; flight_id < n_flights; ++flight_id) {
          const T _left = static_cast<T>(-1.0);
          const T _right = sprite_width_T[flight_id];
          const T _top = static_cast<T>(-1.0);
          const T _bottom = sprite_height_T[flight_id];

          const int sbs = scales_batch_stride[flight_id];

          const T h = sprite_height_T[flight_id];
          const T w = sprite_width_T[flight_id];

          for (int sprite_id = 0; sprite_id < n_sprites[flight_id]; ++sprite_id) {
            const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
            const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];

            const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
            const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];

            const T left = -0.5 + img_width_T * ((_left + 0.5) * scale_x / w + offset_x);
            const T right = -0.5 + img_width_T * ((_right + 0.5) * scale_x / w + offset_x);

            const T top = -0.5 + img_height_T * ((_top + 0.5) * scale_y / h + offset_y);
            const T bottom = -0.5 + img_height_T * ((_bottom + 0.5) * scale_y / h + offset_y);

            const int left_i = static_cast<int>(fmax(0.0, ceil(left)));
            const int right_i = static_cast<int>(fmin(img_width_T-1, floor(right)));

            const int top_i = static_cast<int>(fmax(0.0, ceil(top)));
            const int bottom_i = static_cast<int>(fmin(img_height_T-1, floor(bottom)));

            if (left_i <= right_i && top_i <= bottom_i) {
              for (int i = top_i; i <= bottom_i; i++) {
                for (int j = left_i; j <= right_i; j++) {
                  affecting[i * img_width + j].push_back(std::make_pair(flight_id, sprite_id));
                }
              }
            }
          }
        }

        // --- for each pixel, iterate over all affecting sprites ---

        for (int img_y = 0; img_y < img_height; ++img_y) {
          const T img_y_T = static_cast<T>(img_y);

          for (int img_x = 0; img_x < img_width; ++img_x) {
            const T img_x_T = static_cast<T>(img_x);

            for(int chan = 0; chan < n_channels; ++chan){
                weighted_sum[chan] = 0.0;
                bg[chan] = backgrounds[batch_id * img_batch_stride +
                                       img_y * img_row_stride +
                                       img_x * n_channels + chan];
            }

            T importance_sum = 0.0;
            int n_writes = affecting[img_y * img_width + img_x].size();

            for (auto& idx_pair : affecting[img_y * img_width + img_x]) {
              const int flight_id = idx_pair.first;
              const int sprite_id = idx_pair.second;

              const int sbs = scales_batch_stride[flight_id];

              const T h = sprite_height_T[flight_id];
              const T w = sprite_width_T[flight_id];

              const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
              const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];

              const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
              const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];

              // The pixel location represented in the sprites's co-ordinate frame
              const T y = -0.5 + h * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
              const T x = -0.5 + w * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

              const int fx = std::floor(static_cast<float>(x));
              const int fy = std::floor(static_cast<float>(y));

              const int cx = fx + 1;
              const int cy = fy + 1;

              const T dx = static_cast<T>(cx) - x;
              const T dy = static_cast<T>(cy) - y;

              const T alpha_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels);
              const T alpha_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels);
              const T alpha_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels);
              const T alpha_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels);
              const T alpha = dx * dy * alpha_fxfy +
                              (one - dx) * (one - dy) * alpha_cxcy +
                              dx * (one - dy) * alpha_fxcy +
                              (one - dx) * dy * alpha_cxfy;

              const T imp_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels+1);
              const T imp_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels+1);
              const T imp_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels+1);
              const T imp_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels+1);
              const T imp = dx * dy * imp_fxfy +
                            (one - dx) * (one - dy) * imp_cxcy +
                            dx * (one - dy) * imp_fxcy +
                            (one - dx) * dy * imp_cxfy;

              importance_sum += imp;

              for (int chan = 0; chan < n_channels; ++chan) {
                const T img_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, chan, bg[chan]);
                const T img_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, chan, bg[chan]);
                const T img_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, chan, bg[chan]);
                const T img_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, chan, bg[chan]);
                const T interp = dx * dy * img_fxfy +
                                 (one - dx) * (one - dy) * img_cxcy +
                                 dx * (one - dy) * img_fxcy +
                                 (one - dx) * dy * img_cxfy;

                const T value = alpha * interp + (1-alpha) * bg[chan];
                weighted_sum[chan] += imp * value;
                last_value[chan] = value;
              } // channel
            } // idx_pair

            if(n_writes == 0){
              for (int chan = 0; chan < n_channels; ++chan) {
                const T go = grad_output[batch_id * img_batch_stride +
                                         img_y * img_row_stride +
                                         img_x * n_channels + chan];
                grad_backgrounds[batch_id * img_batch_stride +
                                 img_y * img_row_stride +
                                 img_x * n_channels + chan] = go;
              }
            }else if(n_writes == 1){
              const auto& idx_pair = affecting[img_y * img_width + img_x][0];

              const int flight_id = idx_pair.first;
              const int sprite_id = idx_pair.second;

              const int sbs = scales_batch_stride[flight_id];

              const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
              const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];

              const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
              const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];

              const T y = -0.5 + sprite_height_T[flight_id] * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
              const T x = -0.5 + sprite_width_T[flight_id] * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

              const int fx = std::floor(static_cast<float>(x));
              const int fy = std::floor(static_cast<float>(y));

              const int cx = fx + 1;
              const int cy = fy + 1;

              const T dx = static_cast<T>(cx) - x;
              const T dy = static_cast<T>(cy) - y;

              const T alpha_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels);
              const T alpha_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels);
              const T alpha_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels);
              const T alpha_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels);

              const T alpha = dx * dy * alpha_fxfy +
                              (one - dx) * (one - dy) * alpha_cxcy +
                              dx * (one - dy) * alpha_fxcy +
                              (one - dx) * dy * alpha_cxfy;

              const T alpha_y_factor = dx * (alpha_fxcy - alpha_fxfy) + (1 - dx) * (alpha_cxcy - alpha_cxfy);
              const T alpha_x_factor = dy * (alpha_cxfy - alpha_fxfy) + (1 - dy) * (alpha_cxcy - alpha_fxcy);

              const T grad_y_wrt_scale_y = -sprite_height_T[flight_id] * ((img_y_T + 0.5) / img_height_T - offset_y) / (scale_y * scale_y);
              const T grad_x_wrt_scale_x = -sprite_width_T[flight_id] * ((img_x_T + 0.5) / img_width_T - offset_x) / (scale_x * scale_x);

              const T grad_y_wrt_offset_y = -sprite_height_T[flight_id] / scale_y;
              const T grad_x_wrt_offset_x = -sprite_width_T[flight_id] / scale_x;

              for (int chan = 0; chan < n_channels; ++chan) {
                const T go = grad_output[batch_id * img_batch_stride +
                                         img_y * img_row_stride +
                                         img_x * n_channels + chan];

                grad_backgrounds[batch_id * img_batch_stride +
                                 img_y * img_row_stride +
                                 img_x * n_channels + chan] = go * (1-alpha);

                const T img_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, chan, bg[chan]);
                const T img_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, chan, bg[chan]);
                const T img_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, chan, bg[chan]);
                const T img_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, chan, bg[chan]);

                const T interp = dx * dy * img_fxfy +
                                 (one - dx) * (one - dy) * img_cxcy +
                                 dx * (one - dy) * img_fxcy +
                                 (one - dx) * dy * img_cxfy;

                // ------ update gradient through alpha ------

                const T alpha_premult = go * (interp - bg[chan]);

                update_grad_scales_y(flight_id, batch_id, sprite_id, alpha_premult * alpha_y_factor * grad_y_wrt_scale_y);
                update_grad_scales_x(flight_id, batch_id, sprite_id, alpha_premult * alpha_x_factor * grad_x_wrt_scale_x);

                update_grad_offsets_y(flight_id, batch_id, sprite_id, alpha_premult * alpha_y_factor * grad_y_wrt_offset_y);
                update_grad_offsets_x(flight_id, batch_id, sprite_id, alpha_premult * alpha_x_factor * grad_x_wrt_offset_x);

                update_grad_sprites(flight_id, batch_id, sprite_id, fx, fy, n_channels, alpha_premult * dx * dy);
                update_grad_sprites(flight_id, batch_id, sprite_id, cx, cy, n_channels, alpha_premult * (1-dx) * (1-dy));
                update_grad_sprites(flight_id, batch_id, sprite_id, fx, cy, n_channels, alpha_premult * dx * (1-dy));
                update_grad_sprites(flight_id, batch_id, sprite_id, cx, fy, n_channels, alpha_premult * (1-dx) * dy);

                // ------ update gradient through sprites ------

                const T sprite_premult = go * alpha;

                const T y_factor = dx * (img_fxcy - img_fxfy) + (1 - dx) * (img_cxcy - img_cxfy);
                const T x_factor = dy * (img_cxfy - img_fxfy) + (1 - dy) * (img_cxcy - img_fxcy);

                update_grad_scales_y(flight_id, batch_id, sprite_id, sprite_premult * y_factor * grad_y_wrt_scale_y);
                update_grad_scales_x(flight_id, batch_id, sprite_id, sprite_premult * x_factor * grad_x_wrt_scale_x);

                update_grad_offsets_y(flight_id, batch_id, sprite_id, sprite_premult * y_factor * grad_y_wrt_offset_y);
                update_grad_offsets_x(flight_id, batch_id, sprite_id, sprite_premult * x_factor * grad_x_wrt_offset_x);

                update_grad_sprites(flight_id, batch_id, sprite_id, fx, fy, chan, sprite_premult * dx * dy);
                update_grad_sprites(flight_id, batch_id, sprite_id, cx, cy, chan, sprite_premult * (1-dx) * (1-dy));
                update_grad_sprites(flight_id, batch_id, sprite_id, fx, cy, chan, sprite_premult * dx * (1-dy));
                update_grad_sprites(flight_id, batch_id, sprite_id, cx, fy, chan, sprite_premult * (1-dx) * dy);
              }
            }else{ // n_writes > 1
                T bg_sum = 0.0;
                for (const auto& idx_pair: affecting[img_y * img_width + img_x]) {
                  const int flight_id = idx_pair.first;
                  const int sprite_id = idx_pair.second;

                  const int sbs = scales_batch_stride[flight_id];

                  const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
                  const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];

                  const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
                  const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];

                  const T y = -0.5 + sprite_height_T[flight_id] * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
                  const T x = -0.5 + sprite_width_T[flight_id] * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;

                  const int fx = std::floor(static_cast<float>(x));
                  const int fy = std::floor(static_cast<float>(y));

                  const int cx = fx + 1;
                  const int cy = fy + 1;

                  const T dx = static_cast<T>(cx) - x;
                  const T dy = static_cast<T>(cy) - y;

                  const T alpha_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels);
                  const T alpha_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels);
                  const T alpha_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels);
                  const T alpha_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels);

                  const T alpha = dx * dy * alpha_fxfy +
                                  (one - dx) * (one - dy) * alpha_cxcy +
                                  dx * (one - dy) * alpha_fxcy +
                                  (one - dx) * dy * alpha_cxfy;

                  const T imp_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, n_channels+1);
                  const T imp_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, n_channels+1);
                  const T imp_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, n_channels+1);
                  const T imp_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, n_channels+1);
                  const T imp = dx * dy * imp_fxfy +
                                (one - dx) * (one - dy) * imp_cxcy +
                                dx * (one - dy) * imp_fxcy +
                                (one - dx) * dy * imp_cxfy;

                  bg_sum += imp * (1-alpha);

                  const T alpha_y_factor = dx * (alpha_fxcy - alpha_fxfy) + (1 - dx) * (alpha_cxcy - alpha_cxfy);
                  const T alpha_x_factor = dy * (alpha_cxfy - alpha_fxfy) + (1 - dy) * (alpha_cxcy - alpha_fxcy);

                  const T imp_y_factor = dx * (imp_fxcy - imp_fxfy) + (1 - dx) * (imp_cxcy - imp_cxfy);
                  const T imp_x_factor = dy * (imp_cxfy - imp_fxfy) + (1 - dy) * (imp_cxcy - imp_fxcy);

                  const T grad_y_wrt_scale_y = -sprite_height_T[flight_id] * ((img_y_T + 0.5) / img_height_T - offset_y) / (scale_y * scale_y);
                  const T grad_x_wrt_scale_x = -sprite_width_T[flight_id] * ((img_x_T + 0.5) / img_width_T - offset_x) / (scale_x * scale_x);

                  const T grad_y_wrt_offset_y = -sprite_height_T[flight_id] / scale_y;
                  const T grad_x_wrt_offset_x = -sprite_width_T[flight_id] / scale_x;

                  for (int chan = 0; chan < n_channels; ++chan) {
                    const T go = grad_output[batch_id * img_batch_stride +
                                             img_y * img_row_stride +
                                             img_x * n_channels + chan];

                    const T img_fxfy = get_sprite_data(flight_id, batch_id, sprite_id, fx, fy, chan, bg[chan]);
                    const T img_cxcy = get_sprite_data(flight_id, batch_id, sprite_id, cx, cy, chan, bg[chan]);
                    const T img_fxcy = get_sprite_data(flight_id, batch_id, sprite_id, fx, cy, chan, bg[chan]);
                    const T img_cxfy = get_sprite_data(flight_id, batch_id, sprite_id, cx, fy, chan, bg[chan]);

                    const T interp = dx * dy * img_fxfy +
                                     (one - dx) * (one - dy) * img_cxcy +
                                     dx * (one - dy) * img_fxcy +
                                     (one - dx) * dy * img_cxfy;

                    const T value = alpha * interp + (1-alpha) * bg[chan];

                    // ------ update gradient through alpha ------

                    const T alpha_premult = go * (interp - bg[chan]) * (imp / importance_sum);

                    update_grad_scales_y(flight_id, batch_id, sprite_id, alpha_premult * alpha_y_factor * grad_y_wrt_scale_y);
                    update_grad_scales_x(flight_id, batch_id, sprite_id, alpha_premult * alpha_x_factor * grad_x_wrt_scale_x);

                    update_grad_offsets_y(flight_id, batch_id, sprite_id, alpha_premult * alpha_y_factor * grad_y_wrt_offset_y);
                    update_grad_offsets_x(flight_id, batch_id, sprite_id, alpha_premult * alpha_x_factor * grad_x_wrt_offset_x);

                    update_grad_sprites(flight_id, batch_id, sprite_id, fx, fy, n_channels, alpha_premult * dx * dy);
                    update_grad_sprites(flight_id, batch_id, sprite_id, cx, cy, n_channels, alpha_premult * (1-dx) * (1-dy));
                    update_grad_sprites(flight_id, batch_id, sprite_id, fx, cy, n_channels, alpha_premult * dx * (1-dy));
                    update_grad_sprites(flight_id, batch_id, sprite_id, cx, fy, n_channels, alpha_premult * (1-dx) * dy);

                    // ------ update gradient through imp ------

                    const T imp_premult = go * (value / importance_sum - weighted_sum[chan] / (importance_sum * importance_sum));

                    update_grad_scales_y(flight_id, batch_id, sprite_id, imp_premult * imp_y_factor * grad_y_wrt_scale_y);
                    update_grad_scales_x(flight_id, batch_id, sprite_id, imp_premult * imp_x_factor * grad_x_wrt_scale_x);

                    update_grad_offsets_y(flight_id, batch_id, sprite_id, imp_premult * imp_y_factor * grad_y_wrt_offset_y);
                    update_grad_offsets_x(flight_id, batch_id, sprite_id, imp_premult * imp_x_factor * grad_x_wrt_offset_x);

                    update_grad_sprites(flight_id, batch_id, sprite_id, fx, fy, n_channels+1, imp_premult * dx * dy);
                    update_grad_sprites(flight_id, batch_id, sprite_id, cx, cy, n_channels+1, imp_premult * (1-dx) * (1-dy));
                    update_grad_sprites(flight_id, batch_id, sprite_id, fx, cy, n_channels+1, imp_premult * dx * (1-dy));
                    update_grad_sprites(flight_id, batch_id, sprite_id, cx, fy, n_channels+1, imp_premult * (1-dx) * dy);

                    // ------ update gradient through sprites ------

                    const T sprite_premult = go * alpha * (imp / importance_sum);

                    const T y_factor = dx * (img_fxcy - img_fxfy) + (1 - dx) * (img_cxcy - img_cxfy);
                    const T x_factor = dy * (img_cxfy - img_fxfy) + (1 - dy) * (img_cxcy - img_fxcy);

                    update_grad_scales_y(flight_id, batch_id, sprite_id, sprite_premult * y_factor * grad_y_wrt_scale_y);
                    update_grad_scales_x(flight_id, batch_id, sprite_id, sprite_premult * x_factor * grad_x_wrt_scale_x);

                    update_grad_offsets_y(flight_id, batch_id, sprite_id, sprite_premult * y_factor * grad_y_wrt_offset_y);
                    update_grad_offsets_x(flight_id, batch_id, sprite_id, sprite_premult * x_factor * grad_x_wrt_offset_x);

                    update_grad_sprites(flight_id, batch_id, sprite_id, fx, fy, chan, sprite_premult * dx * dy);
                    update_grad_sprites(flight_id, batch_id, sprite_id, cx, cy, chan, sprite_premult * (1-dx) * (1-dy));
                    update_grad_sprites(flight_id, batch_id, sprite_id, fx, cy, chan, sprite_premult * dx * (1-dy));
                    update_grad_sprites(flight_id, batch_id, sprite_id, cx, fy, chan, sprite_premult * (1-dx) * dy);
                  } // channel
                } // sprite_id - second pass

                for (int chan = 0; chan < n_channels; ++chan) {
                  const T go = grad_output[batch_id * img_batch_stride +
                                           img_y * img_row_stride +
                                           img_x * n_channels + chan];
                  grad_backgrounds[batch_id * img_batch_stride +
                                   img_y * img_row_stride +
                                   img_x * n_channels + chan] = go * bg_sum / importance_sum;
                }
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
    const int64 cost = n_sprites_per_batch * img_height * img_width * n_channels * 1000;

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());

    ::tensorflow::Shard(worker_threads.num_threads,
                        worker_threads.workers,
                        batch_size,
                        cost,
                        update_grads_for_batches);
  }
};

} // namespace functor


template <typename Device, typename T>
class RenderSpritesGradOp : public ::tensorflow::OpKernel {
 public:
  explicit RenderSpritesGradOp(::tensorflow::OpKernelConstruction* context) :
      ::tensorflow::OpKernel(context) {}

  void Compute(::tensorflow::OpKernelContext* ctx) override {

    // --- load lists ---

    OpInputList sprites_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("sprites", &sprites_list));

    const int N = sprites_list.size();

    OP_REQUIRES(ctx, N > 0, ::tensorflow::errors::InvalidArgument("List of sprites tensors must be non-empty."));

    OpInputList scales_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("scales", &scales_list));
    OP_REQUIRES(ctx, scales_list.size() == N,
                ::tensorflow::errors::InvalidArgument("scales_list must have same length as sprites_list, but "
                                                      "the two lengths are ", scales_list.size(), " and ", N));

    OpInputList offsets_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("offsets", &offsets_list));
    OP_REQUIRES(ctx, offsets_list.size() == N,
                ::tensorflow::errors::InvalidArgument("offsets_list must have same length as sprites_list, but "
                                                      "the two lengths are ", offsets_list.size(), " and ", N));

    // --- background ---

    const ::tensorflow::Tensor& backgrounds = ctx->input(3*N);
    const ::tensorflow::TensorShape& bg_shape = backgrounds.shape();

    const int batch_size = bg_shape.dim_size(0);
    const int img_height = bg_shape.dim_size(1);
    const int img_width = bg_shape.dim_size(2);
    const int n_channels = bg_shape.dim_size(3);

    // --- store lists and checks dims ---

    std::vector<const T*> sprites, scales, offsets;
    sprites.reserve(N);
    scales.reserve(N);
    offsets.reserve(N);

    std::vector<int> sprite_shapes;
    sprite_shapes.reserve(4*N);

    for (int i = 0; i < N; ++i) {
      const auto& _sprites = sprites_list[i];
      const auto& _scales = scales_list[i];
      const auto& _offsets = offsets_list[i];

      const ::tensorflow::TensorShape& sprites_shape = _sprites.shape();
      const ::tensorflow::TensorShape& scales_shape = _scales.shape();
      const ::tensorflow::TensorShape& offsets_shape = _offsets.shape();

      // --- check n-dims ---

      OP_REQUIRES(ctx, _sprites.dims() == 5,
                  ::tensorflow::errors::InvalidArgument(
                      "sprites tensor for flight ", i, " must have 5 dimensions, "
                      "but shape is: ", sprites_shape.DebugString()));

      OP_REQUIRES(ctx, _scales.dims() == 3,
                  ::tensorflow::errors::InvalidArgument(
                      "scales tensor for flight ", i, " must have 3 dimensions, "
                      "but shape is: ", scales_shape.DebugString()));

      OP_REQUIRES(ctx, _offsets.dims() == 3,
                  ::tensorflow::errors::InvalidArgument(
                      "offsets tensor for flight ", i, " must have 3 dimensions, "
                      "but shape is: ", offsets_shape.DebugString()));

      // ------ check batch size ------
      //
      OP_REQUIRES(ctx, sprites_shape.dim_size(0) == batch_size,
                  ::tensorflow::errors::InvalidArgument(
                      "Batch size for sprites in flight ", i, " must equal batch size for background tensor, "
                      "but shapes are: ", sprites_shape.DebugString(), ", ",
                      bg_shape.DebugString()));

      OP_REQUIRES(ctx, scales_shape.dim_size(0) == batch_size,
                  ::tensorflow::errors::InvalidArgument(
                      "Batch size for scales in flight ", i, " must equal batch size for background tensor, "
                      "but shapes are: ", scales_shape.DebugString(), ", ",
                      bg_shape.DebugString()));

      OP_REQUIRES(ctx, offsets_shape.dim_size(0) == batch_size,
                  ::tensorflow::errors::InvalidArgument(
                      "Batch size for offsets in flight ", i, " must equal batch size for background tensor, "
                      "but shapes are: ", offsets_shape.DebugString(), ", ",
                      bg_shape.DebugString()));

      // ------ check n_sprites ------

      const int n_sprites = sprites_shape.dim_size(1);

      OP_REQUIRES(ctx, scales_shape.dim_size(1) == n_sprites,
                  ::tensorflow::errors::InvalidArgument(
                      "Max sprites (dim 1) of sprites and scales tensor must be the "
                      "same, but shapes for flight ", i, " are: ",
                      sprites_shape.DebugString(), ", ", scales_shape.DebugString()));

      OP_REQUIRES(ctx, offsets_shape.dim_size(1) == n_sprites,
                  ::tensorflow::errors::InvalidArgument(
                      "Max sprites (dim 1) of sprites and offsets tensor must be the "
                      "same, but shapes for flight ", i, " are: ",
                      sprites_shape.DebugString(), ", ", offsets_shape.DebugString()));

      // ------ trailing dims ------

      // because sprites have an alpha and importance channels
      OP_REQUIRES(ctx, sprites_shape.dim_size(4) == n_channels + 2,
                  ::tensorflow::errors::InvalidArgument(
                      "Trailing dimension of sprites must be n_channels + 2, "
                      "but shape for flight ", i, " is : ", sprites_shape.DebugString(),
                      " while n_channels is ", n_channels));

      OP_REQUIRES(ctx, scales_shape.dim_size(2) == 2,
                  ::tensorflow::errors::InvalidArgument(
                      "Trailing dimension of scales must be 2, "
                      "but shape for flight ", i, " is : ", scales_shape.DebugString()));

      OP_REQUIRES(ctx, offsets_shape.dim_size(2) == 2,
                  ::tensorflow::errors::InvalidArgument(
                      "Trailing dimension of offsets must be 2, "
                      "but shape for flight ", i, " is : ", offsets_shape.DebugString()));

      sprites.push_back(_sprites.flat<T>().data());
      scales.push_back(_scales.flat<T>().data());
      offsets.push_back(_offsets.flat<T>().data());

      sprite_shapes.push_back(sprites_shape.dim_size(1));
      sprite_shapes.push_back(sprites_shape.dim_size(2));
      sprite_shapes.push_back(sprites_shape.dim_size(3));
      sprite_shapes.push_back(sprites_shape.dim_size(4));
    }

    const ::tensorflow::Tensor& grad_output = ctx->input(3*N+1);
    const ::tensorflow::TensorShape& grad_output_shape = grad_output.shape();

    OP_REQUIRES(ctx, grad_output_shape == bg_shape,
                ::tensorflow::errors::InvalidArgument(
                   "grad_output shape is not consistent with backgrounds "
                   "shape; it should be ",
                   bg_shape.DebugString(), " but is ",
                   grad_output_shape.DebugString()));

    // --- Allocate outputs ---

    OpOutputList grads_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("grads", &grads_list));

    ::tensorflow::Tensor* grad_backgrounds = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(3*N, backgrounds.shape(), &grad_backgrounds));

    std::vector<T*> grad_sprites, grad_scales, grad_offsets;

    for (int i = 0; i < N; ++i) {
      const auto& _sprites = sprites_list[i];
      const auto& _scales = scales_list[i];
      const auto& _offsets = offsets_list[i];

      const ::tensorflow::TensorShape& sprites_shape = _sprites.shape();
      const ::tensorflow::TensorShape& scales_shape = _scales.shape();
      const ::tensorflow::TensorShape& offsets_shape = _offsets.shape();

      Tensor *_grad_sprites, *_grad_scales, *_grad_offsets;
      OP_REQUIRES_OK(ctx, grads_list.allocate(i, sprites_shape, &_grad_sprites));
      OP_REQUIRES_OK(ctx, grads_list.allocate(N+i, scales_shape, &_grad_scales));
      OP_REQUIRES_OK(ctx, grads_list.allocate(2*N+i, offsets_shape, &_grad_offsets));

      grad_sprites.push_back(_grad_sprites->flat<T>().data());
      grad_scales.push_back(_grad_scales->flat<T>().data());
      grad_offsets.push_back(_grad_offsets->flat<T>().data());
    }

    // OpOutputList grad_sprites_list, grad_scales_list, grad_offsets_list;
    // OP_REQUIRES_OK(ctx, ctx->output_list("grad_sprites", &grad_sprites_list));
    // OP_REQUIRES_OK(ctx, ctx->output_list("grad_scales", &grad_scales_list));
    // OP_REQUIRES_OK(ctx, ctx->output_list("grad_offsets", &grad_offsets_list));

    // ::tensorflow::Tensor* grad_backgrounds = nullptr;
    // OP_REQUIRES_OK(ctx, ctx->allocate_output(3, backgrounds.shape(), &grad_backgrounds));

    // std::vector<T*> grad_sprites, grad_scales, grad_offsets;

    // for (int i = 0; i < N; ++i) {
    //   const auto& _sprites = sprites_list[i];
    //   const auto& _scales = scales_list[i];
    //   const auto& _offsets = offsets_list[i];

    //   const ::tensorflow::TensorShape& sprites_shape = _sprites.shape();
    //   const ::tensorflow::TensorShape& scales_shape = _scales.shape();
    //   const ::tensorflow::TensorShape& offsets_shape = _offsets.shape();

    //   Tensor *_grad_sprites, *_grad_scales, *_grad_offsets;
    //   OP_REQUIRES_OK(ctx, grad_sprites_list.allocate(i, sprites_shape, &_grad_sprites));
    //   OP_REQUIRES_OK(ctx, grad_scales_list.allocate(i, scales_shape, &_grad_scales));
    //   OP_REQUIRES_OK(ctx, grad_offsets_list.allocate(i, offsets_shape, &_grad_offsets));

    //   grad_sprites.push_back(_grad_sprites->flat<T>().data());
    //   grad_scales.push_back(_grad_scales->flat<T>().data());
    //   grad_offsets.push_back(_grad_offsets->flat<T>().data());
    // }

    functor::RenderSpritesGrad2DFunctor<Device, T>()(ctx,
                                                     ctx->eigen_device<Device>(),

                                                     sprite_shapes,

                                                     sprites,
                                                     scales,
                                                     offsets,
                                                     backgrounds.flat<T>().data(),

                                                     grad_output.flat<T>().data(),

                                                     grad_sprites,
                                                     grad_scales,
                                                     grad_offsets,
                                                     grad_backgrounds->flat<T>().data(),

                                                     batch_size,
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
