// Copyright 2016 The Sonnet Authors. All Rights Reserved.
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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "render_sprites_ops.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cmath>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MAX
#define MAX(x,y) ((x < y) ? y : x)
#endif

using GPUDevice = Eigen::GpuDevice;

struct FlightInfo {
    int n_sprites;
    int sprite_height;
    int sprite_width;
    int sprites_batch_stride;
    int sprites_sprite_stride;
    int sprites_row_stride;
    int scales_batch_stride;
};

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void getNumBlocksAndThreads(const GPUDevice &d, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    // get device capability, to avoid block/grid size exceed the upper bound
    // const cudaDeviceProp &prop = d.stream()->deviceProperties();
    // if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    // {
    //     printf("n is too large, please choose a smaller number!\n");
    // }

    // if (blocks > prop.maxGridSize[0])
    // {
    //     printf("Grid size <%d> exceeds the device capability <%d>, set block size as %d (original %d)\n",
    //            blocks, prop.maxGridSize[0], threads*2, threads);

    //     blocks /= 2;
    //     threads *= 2;
    // }

    blocks = MIN(maxBlocks, blocks);
}

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize, bool nIsPow2>
__global__ void
reduce_max_kernel(int *g_idata, int *g_odata, unsigned int n)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int *sdata = SharedMemory<int>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    int myMax = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        myMax = MAX(myMax, g_idata[i]);

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            myMax = MAX(myMax, g_idata[i+blockSize]);

        i += gridSize;
    }

    // each thread puts its local max into shared memory
    sdata[tid] = myMax;
    cg::sync(cta);

    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = myMax = MAX(myMax, sdata[tid + 256]);
    }

    cg::sync(cta);

    if ((blockSize >= 256) &&(tid < 128))
    {
        sdata[tid] = myMax = MAX(myMax, sdata[tid + 128]);
    }

    cg::sync(cta);

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 64]);
    }

    cg::sync(cta);

    if ((blockSize >=  64) && (tid < 32))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 32]);
    }

    cg::sync(cta);

    if ((blockSize >=  32) && (tid < 16))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 16]);
    }

    cg::sync(cta);

    if ((blockSize >=  16) && (tid <  8))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 8]);
    }

    cg::sync(cta);

    if ((blockSize >=   8) && (tid <  4))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 4]);
    }

    cg::sync(cta);

    if ((blockSize >=   4) && (tid <  2))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 2]);
    }

    cg::sync(cta);

    if ((blockSize >=   2) && ( tid <  1))
    {
       sdata[tid] = myMax = MAX(myMax, sdata[tid + 1]);
    }

    cg::sync(cta);

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = myMax;
}

void _reduce_max(int size, int threads, int blocks, int *d_idata, int *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);

    if (isPow2(size)) {
        switch (threads)
        {
            case 512:
                reduce_max_kernel<512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce_max_kernel<256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce_max_kernel<128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce_max_kernel<64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce_max_kernel<32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce_max_kernel<16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduce_max_kernel<8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduce_max_kernel<4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduce_max_kernel<2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduce_max_kernel<1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    } else {
        switch (threads)
        {
            case 512:
                reduce_max_kernel<512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 256:
                reduce_max_kernel<256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 128:
                reduce_max_kernel<128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 64:
                reduce_max_kernel<64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 32:
                reduce_max_kernel<32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case 16:
                reduce_max_kernel<16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  8:
                reduce_max_kernel<8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  4:
                reduce_max_kernel<4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  2:
                reduce_max_kernel<2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;

            case  1:
                reduce_max_kernel<1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
                break;
        }
    }
}

int reduce_max(const GPUDevice& d, int n, int maxThreads, int maxBlocks, int *d_idata, int *d_odata)
{
    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(d, n, maxBlocks, maxThreads, numBlocks, numThreads);

    d.synchronize();

    _reduce_max(n, numThreads, numBlocks, d_idata, d_odata);

    int s = numBlocks;

    while (s > 1)
    {
        getNumBlocksAndThreads(d, s, maxBlocks, maxThreads, numBlocks, numThreads);
        d.memcpy(d_idata, d_odata, s*sizeof(int));
        _reduce_max(s, numThreads, numBlocks, d_idata, d_odata);
        s = (s + (numThreads*2-1)) / (numThreads*2);
    }

    d.synchronize();

    int result = 0;
    d.memcpyDeviceToHost(&result, d_odata, sizeof(int));
    return result;
}

template <typename T>
__global__ void index_affecting_sprites(const T *const * __restrict__ scales,
                                        const T *const * __restrict__ offsets,

                                        int* __restrict__ pairs,
                                        int* __restrict__ affecting,
                                        int* __restrict__ counts,

                                        FlightInfo* __restrict__ flight_info,

                                        const int total_n_sprites,
                                        const int n_sprites_per_batch,

                                        const int img_height,
                                        const int img_width,
                                        const int max_affecting){

  // For each pixel, index the affecting sprites

  const T img_height_T = static_cast<T>(img_height);
  const T img_width_T = static_cast<T>(img_width);

  const T _left = static_cast<T>(-1.0);
  const T _top = static_cast<T>(-1.0);

  CUDA_1D_KERNEL_LOOP(index, total_n_sprites) {
    const int batch_id = index / n_sprites_per_batch;
    const int pair_id = index % n_sprites_per_batch;

    const int flight_id = pairs[2*pair_id];
    const int sprite_id = pairs[2*pair_id+1];

    int sbs = flight_info[flight_id].scales_batch_stride;

    const T scale_y = scales[flight_id][batch_id * sbs + sprite_id * 2];
    const T scale_x = scales[flight_id][batch_id * sbs + sprite_id * 2 + 1];
    const T offset_y = offsets[flight_id][batch_id * sbs + sprite_id * 2];
    const T offset_x = offsets[flight_id][batch_id * sbs + sprite_id * 2 + 1];
    const T h = static_cast<T>(flight_info[flight_id].sprite_height);
    const T w = static_cast<T>(flight_info[flight_id].sprite_width);

    const T _right = w;
    const T _bottom = h;

    const T left = -0.5 + w * ((_left + 0.5) * scale_x / w + offset_x);
    const T right = -0.5 + w * ((_right + 0.5) * scale_x / w + offset_x);
    const T top = -0.5 + h * ((_top + 0.5) * scale_y / h + offset_y);
    const T bottom = -0.5 + h * ((_bottom + 0.5) * scale_y / h + offset_y);

    const int left_i = static_cast<int>(MAX(0.0, ceil(left)));
    const int right_i = static_cast<int>(MIN(img_width_T-1, floor(right)));

    const int top_i = static_cast<int>(MAX(0.0, ceil(top)));
    const int bottom_i = static_cast<int>(MIN(img_height_T-1, floor(bottom)));

    if (left_i <= right_i && top_i <= bottom_i) {
      for (int i = top_i; i <= bottom_i; i++) {
        for (int j = left_i; j <= right_i; j++) {
          int count = atomicAdd(
            counts
            + batch_id * img_height * img_width
            + i * img_width
            + j,
            1
          );

          if (max_affecting >= 0) {
            affecting[
              batch_id * img_height * img_width * max_affecting
              + i * img_width * max_affecting
              + j * max_affecting
              + count] = pair_id;
          }
        }
      }
    }
  }
}

namespace tensorflow {

namespace {

#define GET_SPRITE_POINT(x, y) \
  _sprites[batch_id * sbs + \
           sprite_id * sss + \
           y * srs + \
           x * (n_channels + 2) + \
           chan]

#define GET_ALPHA_POINT(x, y) \
  _sprites[batch_id * sbs + \
           sprite_id * sss + \
           y * srs + \
           x * (n_channels + 2) + \
           n_channels]

#define GET_IMPORTANCE_POINT(x, y) \
  _sprites[batch_id * sbs + \
           sprite_id * sss + \
           y * srs + \
           x * (n_channels + 2) + \
           n_channels + 1]

#define FORWARD_PASS_LOOP_BODY \
      const int pair_id = affecting[pixel_id * max_affecting + i]; \
      const int flight_id = pairs[2*pair_id]; \
      const int sprite_id = pairs[2*pair_id+1]; \
 \
      const int scale_bs = flight_info[flight_id].scales_batch_stride; \
 \
      const T scale_y = scales[flight_id][batch_id * scale_bs + sprite_id * 2]; \
      const T scale_x = scales[flight_id][batch_id * scale_bs + sprite_id * 2 + 1]; \
 \
      const T offset_y = offsets[flight_id][batch_id * scale_bs + sprite_id * 2]; \
      const T offset_x = offsets[flight_id][batch_id * scale_bs + sprite_id * 2 + 1]; \
 \
      const int h = flight_info[flight_id].sprite_height; \
      const int w = flight_info[flight_id].sprite_width; \
      const int h_T = static_cast<T>(h); \
      const int w_T = static_cast<T>(w); \
 \
      const T y = -0.5 + h_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y; \
      const T x = -0.5 + w_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x; \
 \
      const int fx = std::floor(static_cast<float>(x)); \
      const int fy = std::floor(static_cast<float>(y)); \
 \
      const int cx = fx + 1; \
      const int cy = fy + 1; \
 \
      const T dx = static_cast<T>(cx) - x; \
      const T dy = static_cast<T>(cy) - y; \
 \
      const T* _sprites = sprites[flight_id]; \
      const int sbs = flight_info[flight_id].sprites_batch_stride; \
      const int sss = flight_info[flight_id].sprites_sprite_stride; \
      const int srs = flight_info[flight_id].sprites_row_stride; \
 \
      const T alpha_fxfy = (fx >= 0 && fy >= 0) \
                         ? GET_ALPHA_POINT(fx, fy) \
                         : zero; \
      const T alpha_cxcy = (cx <= w - 1 && cy <= h - 1) \
                         ? GET_ALPHA_POINT(cx, cy) \
                         : zero; \
      const T alpha_fxcy = (fx >= 0 && cy <= h - 1) \
                         ? GET_ALPHA_POINT(fx, cy) \
                         : zero; \
      const T alpha_cxfy = (cx <= w - 1 && fy >= 0) \
                         ? GET_ALPHA_POINT(cx, fy) \
                         : zero; \
      const T alpha = dx * dy * alpha_fxfy + \
                      (one - dx) * (one - dy) * alpha_cxcy + \
                      dx * (one - dy) * alpha_fxcy + \
                      (one - dx) * dy * alpha_cxfy; \
 \
      const T importance_fxfy = (fx >= 0 && fy >= 0) \
                         ? GET_IMPORTANCE_POINT(fx, fy) \
                         : zero; \
      const T importance_cxcy = (cx <= w - 1 && cy <= h - 1) \
                         ? GET_IMPORTANCE_POINT(cx, cy) \
                         : zero; \
      const T importance_fxcy = (fx >= 0 && cy <= h - 1) \
                         ? GET_IMPORTANCE_POINT(fx, cy) \
                         : zero; \
      const T importance_cxfy = (cx <= w - 1 && fy >= 0) \
                         ? GET_IMPORTANCE_POINT(cx, fy) \
                         : zero; \
      const T importance = dx * dy * importance_fxfy + \
                           (one - dx) * (one - dy) * importance_cxcy + \
                           dx * (one - dy) * importance_fxcy + \
                           (one - dx) * dy * importance_cxfy; \
 \
      const T img_fxfy = (fx >= 0 && fy >= 0) \
                         ? GET_SPRITE_POINT(fx, fy) \
                         : bg; \
      const T img_cxcy = (cx <= w - 1 && cy <= h - 1) \
                         ? GET_SPRITE_POINT(cx, cy) \
                         : bg; \
      const T img_fxcy = (fx >= 0 && cy <= h - 1) \
                         ? GET_SPRITE_POINT(fx, cy) \
                         : bg; \
      const T img_cxfy = (cx <= w - 1 && fy >= 0) \
                         ? GET_SPRITE_POINT(cx, fy) \
                         : bg; \
      const T interp = dx * dy * img_fxfy + \
                       (one - dx) * (one - dy) * img_cxcy + \
                       dx * (one - dy) * img_fxcy + \
                       (one - dx) * dy * img_cxfy;

template <typename T>
__global__ void RenderSprites2DKernel(const T *const * __restrict__ sprites,
                                      const T *const * __restrict__ scales,
                                      const T *const * __restrict__ offsets,
                                      const T* __restrict__ backgrounds,

                                      const int* __restrict__ pairs,
                                      const int* __restrict__ affecting,
                                      const int* __restrict__ counts,

                                      T* __restrict__ output,

                                      FlightInfo* __restrict__ flight_info,

                                      const int batch_size,
                                      const int img_height,
                                      const int img_width,
                                      const int n_channels,
                                      const int max_affecting){

  const int output_size = batch_size * img_height * img_width * n_channels;

  const int img_batch_stride = img_height * img_width * n_channels;
  const int img_row_stride = img_width * n_channels;

  const T img_height_T = static_cast<T>(img_height);
  const T img_width_T = static_cast<T>(img_width);

  const T zero = static_cast<T>(0.0);
  const T one = static_cast<T>(1.0);

  CUDA_1D_KERNEL_LOOP(index, output_size) {
    const int pixel_id = index / n_channels;
    const int batch_id = index / img_batch_stride;
    const int index_in_batch = index % img_batch_stride;
    const int img_y = index_in_batch / img_row_stride;
    const int index_in_row = index_in_batch % img_row_stride;
    const int img_x = index_in_row / n_channels;
    const int chan = index_in_row % n_channels;

    const T img_y_T = static_cast<T>(img_y);
    const T img_x_T = static_cast<T>(img_x);

    T weighted_sum = 0.0;
    T importance_sum = 0.0;
    int n_writes = counts[pixel_id];
    T bg = backgrounds[index];
    T last_value = 0.0;

    for(int i = 0; i < counts[pixel_id]; ++i) {
      FORWARD_PASS_LOOP_BODY

      last_value = alpha * interp + (1-alpha) * bg;
      weighted_sum += importance * last_value;
      importance_sum += importance;
    }

    if(n_writes == 0){
      output[index] = bg;
    }else if(n_writes == 1){
      output[index] = last_value;
    }else{
      output[index] = weighted_sum / importance_sum;
    }
  }
}

}  // namespace

namespace functor {

// modelled after SetZero
template <typename T>
__global__ void Copy(const int count, T* dst, const T* src) {
  // Check that the grid is one dimensional and index doesn't overflow.
  assert(blockDim.y == 1 && blockDim.z == 1);
  assert(blockDim.x * gridDim.x / blockDim.x == gridDim.x);
  for (int i : CudaGridRangeX(count)) {
    dst[i] = src[i];
  }
}

#define FUNCTOR_SETUP \
    const int n_flights = sprites.size(); \
 \
    /* --- allocate on-device arrays for the per-flight pointers to data --- */ \
 \
    Tensor sprites_t; \
    T** sprites_d; \
    allocate_temp_array<T*>(ctx, &sprites_d, n_flights, &sprites_t); \
    cudaMemcpy(sprites_d, &sprites[0], sizeof(T*)*n_flights, cudaMemcpyHostToDevice); \
 \
    Tensor scales_t; \
    T** scales_d; \
    allocate_temp_array<T*>(ctx, &scales_d, scales.size(), &scales_t); \
    cudaMemcpy(scales_d, &scales[0], sizeof(T*)*n_flights, cudaMemcpyHostToDevice); \
 \
    Tensor offsets_t; \
    T** offsets_d; \
    allocate_temp_array<T*>(ctx, &offsets_d, offsets.size(), &offsets_t); \
    cudaMemcpy(offsets_d, &offsets[0], sizeof(T*)*n_flights, cudaMemcpyHostToDevice); \
 \
    /* -- compute flight_info --- */ \
 \
    std::vector<FlightInfo> flight_info; \
    flight_info.reserve(n_flights); \
 \
    int n_sprites_per_batch = 0; \
 \
    for(int i=0; i < n_flights; i++){ \
      const int n = shapes[4*i]; \
      const int h = shapes[4*i+1]; \
      const int w = shapes[4*i+2]; \
      const int c = shapes[4*i+3]; \
 \
      const int sprites_batch_stride = n * h * w * c; \
      const int sprites_sprite_stride = h * w * c; \
      const int sprites_row_stride = w * c; \
      const int scales_batch_stride = 2 * n; \
 \
      FlightInfo _flight_info{n, h, w, sprites_batch_stride, sprites_sprite_stride, sprites_row_stride, scales_batch_stride}; \
      flight_info.push_back(_flight_info); \
 \
      n_sprites_per_batch += n; \
    } \
 \
    Tensor flight_info_tensor; \
    FlightInfo* flight_info_d; \
    allocate_temp_array<FlightInfo>(ctx, &flight_info_d, n_flights, &flight_info_tensor); \
    cudaMemcpy(flight_info_d, &flight_info[0], sizeof(FlightInfo)*n_flights, cudaMemcpyHostToDevice); \
 \
    /* --- compute (flight_id, sprite_id) pairs */ \
 \
    std::vector<int> pairs(n_sprites_per_batch*2); \
    int pair_id = 0; \
    for(int i=0; i < n_flights; i++){ \
      for(int j=0; j < flight_info[i].n_sprites; j++){ \
        pairs[2*pair_id] = i; \
        pairs[2*pair_id+1] = j; \
        pair_id++; \
      } \
    } \
 \
    Tensor pairs_t; \
    int* pairs_d; \
    allocate_temp_array<int>(ctx, &pairs_d, 2*n_sprites_per_batch, &pairs_t); \
    cudaMemcpy(pairs_d, &pairs[0], sizeof(int)*n_sprites_per_batch*2, cudaMemcpyHostToDevice); \
 \
    /* --- */ \
 \
    const int total_n_sprites = batch_size * n_sprites_per_batch; \
    const int total_n_pixels = batch_size * img_height * img_width; \
 \
    Tensor counts_t; \
    int* counts_d; \
    allocate_temp_array<int>(ctx, &counts_d, total_n_pixels, &counts_t); \
 \
    ::tensorflow::CudaLaunchConfig config; \
 \
    /* --- set counts to zero --- */ \
 \
    config = ::tensorflow::GetCudaLaunchConfig(total_n_pixels, d); \
    ::tensorflow::SetZero \
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(total_n_pixels, counts_d); \
 \
    /* --- for each pixel, count number of sprites that affect it --- */ \
 \
    config = ::tensorflow::GetCudaLaunchConfig(total_n_sprites, d); \
    index_affecting_sprites<T> \
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>( \
            scales_d, offsets_d, pairs_d, nullptr, counts_d, flight_info_d, \
            total_n_sprites, n_sprites_per_batch, img_height, img_width, -1); \
 \
    /* --- get max of counts so we know how much space to allocate for array `affecting` --- */ \
 \
    const int maxBlocks = 64; \
    const int maxThreads = 256; \
 \
    Tensor temp_t; \
    int* temp_d; \
    allocate_temp_array<int>(ctx, &temp_d, maxBlocks, &temp_t); \
 \
    const int max_affecting = reduce_max(d, total_n_pixels, maxThreads, maxBlocks, counts_d, temp_d); \
 \
    /* --- set counts to zero again --- */ \
 \
    config = ::tensorflow::GetCudaLaunchConfig(total_n_pixels, d); \
    ::tensorflow::SetZero \
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(total_n_pixels, counts_d); \
 \
    /* --- for each pixel, find the indices of the sprites that affect it --- */ \
 \
    Tensor affecting_t; \
    int* affecting_d; \
    int affecting_size = total_n_pixels * max_affecting; \
    allocate_temp_array<int>(ctx, &affecting_d, affecting_size, &affecting_t); \
 \
    config = ::tensorflow::GetCudaLaunchConfig(total_n_sprites, d); \
    index_affecting_sprites<T> \
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>( \
            scales_d, offsets_d, pairs_d, affecting_d, counts_d, flight_info_d, \
            total_n_sprites, n_sprites_per_batch, img_height, img_width, max_affecting);


template <typename T>
struct RenderSprites2DFunctor<GPUDevice, T>{
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const GPUDevice& d,

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
    FUNCTOR_SETUP

    // --- compute the image ---

    const int output_size = batch_size * img_height * img_width * n_channels;
    config = ::tensorflow::GetCudaLaunchConfig(output_size, d);
    RenderSprites2DKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            sprites_d, scales_d, offsets_d, backgrounds, pairs_d, affecting_d, counts_d, output,
            flight_info_d, batch_size, img_height, img_width, n_channels, max_affecting);
  }
};

// template struct RenderSprites2DFunctor<GPUDevice, Eigen::half>;
template struct RenderSprites2DFunctor<GPUDevice, float>;
// template struct RenderSprites2DFunctor<GPUDevice, double>;

}  // namespace functor




namespace {

#define UPDATE_GRAD_SPRITES(x, y, v) \
  atomicAdd(grad_sprites[flight_id] + \
            batch_id * sbs + \
            sprite_id * sss + \
            y * srs + \
            x * (n_channels + 2) + \
            chan, \
            v)

#define UPDATE_GRAD_ALPHAS(x, y, v) \
  atomicAdd(grad_sprites[flight_id] + \
            batch_id * sbs + \
            sprite_id * sss + \
            y * srs + \
            x * (n_channels + 2) + \
            n_channels, \
            v)

#define UPDATE_GRAD_IMPORTANCES(x, y, v) \
  atomicAdd(grad_sprites[flight_id] + \
            batch_id * sbs + \
            sprite_id * sss + \
            y * srs + \
            x * (n_channels + 2) + \
            n_channels + 1, \
            v)

#define UPDATE_GRAD_SCALES_Y(v) \
  atomicAdd(grad_scales[flight_id] + \
            batch_id * scale_bs + \
            sprite_id * 2, \
            v)

#define UPDATE_GRAD_SCALES_X(v) \
  atomicAdd(grad_scales[flight_id] + \
            batch_id * scale_bs + \
            sprite_id * 2 + 1, \
            v)

#define UPDATE_GRAD_OFFSETS_Y(v) \
  atomicAdd(grad_offsets[flight_id] + \
            batch_id * scale_bs + \
            sprite_id * 2, \
            v)

#define UPDATE_GRAD_OFFSETS_X(v) \
  atomicAdd(grad_offsets[flight_id] + \
            batch_id * scale_bs + \
            sprite_id * 2 + 1, \
            v)

// template <typename T>
// __global__ void RenderSpritesGrad2DKernel(const T* __restrict__ sprites,
//                                           const int* __restrict__ n_sprites,
//                                           const T* __restrict__ scales,
//                                           const T* __restrict__ offsets,
//                                           const T* __restrict__ backgrounds,
// 
//                                           const int* __restrict__ affecting,
//                                           const int* __restrict__ counts,
//                                           const T* __restrict__ grad_output,
// 
//                                           T* __restrict__ grad_sprites,
//                                           T* __restrict__ grad_n_sprites,
//                                           T* __restrict__ grad_scales,
//                                           T* __restrict__ grad_offsets,
//                                           T* __restrict__ grad_backgrounds,
// 
//                                           const int batch_size,
// 
//                                           const int max_sprites,
//                                           const int sprite_height,
//                                           const int sprite_width,
// 
//                                           const int img_height,
//                                           const int img_width,
// 
//                                           const int n_channels,
//                                           const int max_affecting){

//  const int output_size = batch_size * img_height * img_width * n_channels;
//  const int sprites_batch_stride = max_sprites * sprite_height * sprite_width * (n_channels + 2);
//  const int sprites_sprite_stride = sprite_height * sprite_width * (n_channels + 2);
//  const int sprites_row_stride = sprite_width * (n_channels + 2);
//
//  const int scales_batch_stride = 2 * max_sprites;
//  const int offsets_batch_stride = 2 * max_sprites;
//
//  const int img_batch_stride = img_height * img_width * n_channels;
//  const int img_row_stride = img_width * n_channels;
//
//  const T sprite_height_T = static_cast<T>(sprite_height);
//  const T sprite_width_T = static_cast<T>(sprite_width);
//
//  const T img_height_T = static_cast<T>(img_height);
//  const T img_width_T = static_cast<T>(img_width);
//
//  const T zero = static_cast<T>(0.0);
//  const T one = static_cast<T>(1.0);
//
//  CUDA_1D_KERNEL_LOOP(index, output_size) {
//    const int pixel_id = index / n_channels;
//    const int batch_id = index / img_batch_stride;
//    const int index_in_batch = index % img_batch_stride;
//    const int img_y = index_in_batch / img_row_stride;
//    const int index_in_row = index_in_batch % img_row_stride;
//    const int img_x = index_in_row / n_channels;
//    const int chan = index_in_row % n_channels;
//
//    const T img_y_T = static_cast<T>(img_y);
//    const T img_x_T = static_cast<T>(img_x);
//
//    T weighted_sum = 0.0;
//    T importance_sum = 0.0;
//    int n_writes = counts[pixel_id];
//    T bg = backgrounds[index];
//    T bg_sum = 0.0;
//    T alpha = 0.0;
//
//    for(int i = 0; i < counts[pixel_id]; ++i) {
//      const int sprite_id = affecting[pixel_id * max_affecting + i];
//
//      const T scale_y = scales[batch_id * scales_batch_stride + sprite_id * 2];
//      const T scale_x = scales[batch_id * scales_batch_stride + sprite_id * 2 + 1];
//
//      const T offset_y = offsets[batch_id * offsets_batch_stride + sprite_id * 2];
//      const T offset_x = offsets[batch_id * offsets_batch_stride + sprite_id * 2 + 1];
//
//      // The pixel location represented in the sprites's co-ordinate frame
//      const T y = -0.5 + sprite_height_T * ((img_y_T + 0.5) / img_height_T - offset_y) / scale_y;
//      const T x = -0.5 + sprite_width_T * ((img_x_T + 0.5) / img_width_T - offset_x) / scale_x;
//
//      const int fx = std::floor(static_cast<float>(x));
//      const int fy = std::floor(static_cast<float>(y));
//
//      const int cx = fx + 1;
//      const int cy = fy + 1;
//
//      const T dx = static_cast<T>(cx) - x;
//      const T dy = static_cast<T>(cy) - y;
//
//      const T alpha_fxfy = (fx >= 0 && fy >= 0)
//                           ? GET_ALPHA_POINT(fx, fy)
//                           : zero;
//      const T alpha_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
//                           ? GET_ALPHA_POINT(cx, cy)
//                           : zero;
//      const T alpha_fxcy = (fx >= 0 && cy <= sprite_height - 1)
//                           ? GET_ALPHA_POINT(fx, cy)
//                           : zero;
//      const T alpha_cxfy = (cx <= sprite_width - 1 && fy >= 0)
//                           ? GET_ALPHA_POINT(cx, fy)
//                           : zero;
//      const T alpha = dx * dy * alpha_fxfy +
//                      (one - dx) * (one - dy) * alpha_cxcy +
//                      dx * (one - dy) * alpha_fxcy +
//                      (one - dx) * dy * alpha_cxfy;
//
//      const T importance_fxfy = (fx >= 0 && fy >= 0)
//                         ? GET_IMPORTANCE_POINT(fx, fy)
//                         : zero;
//      const T importance_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
//                         ? GET_IMPORTANCE_POINT(cx, cy)
//                         : zero;
//      const T importance_fxcy = (fx >= 0 && cy <= sprite_height - 1)
//                         ? GET_IMPORTANCE_POINT(fx, cy)
//                         : zero;
//      const T importance_cxfy = (cx <= sprite_width - 1 && fy >= 0)
//                         ? GET_IMPORTANCE_POINT(cx, fy)
//                         : zero;
//      const T importance = dx * dy * importance_fxfy +
//                           (one - dx) * (one - dy) * importance_cxcy +
//                           dx * (one - dy) * importance_fxcy +
//                           (one - dx) * dy * importance_cxfy;
//
//      bg_sum += importance * (1-alpha);
//
//      const T img_fxfy = (fx >= 0 && fy >= 0)
//                         ? GET_SPRITE_POINT(fx, fy)
//                         : bg;
//      const T img_cxcy = (cx <= sprite_width - 1 && cy <= sprite_height - 1)
//                         ? GET_SPRITE_POINT(cx, cy)
//                         : bg;
//      const T img_fxcy = (fx >= 0 && cy <= sprite_height - 1)
//                         ? GET_SPRITE_POINT(fx, cy)
//                         : bg;
//      const T img_cxfy = (cx <= sprite_width - 1 && fy >= 0)
//                         ? GET_SPRITE_POINT(cx, fy)
//                         : bg;
//      const T interp = dx * dy * img_fxfy +
//                       (one - dx) * (one - dy) * img_cxcy +
//                       dx * (one - dy) * img_fxcy +
//                       (one - dx) * dy * img_cxfy;
//
//      const T value = alpha * interp + (1-alpha) * bg;
//      weighted_sum += importance * value;
//      importance_sum += importance;
//    } // sprite_id - forward pass


template <typename T>
__global__ void RenderSpritesGrad2DKernel(const T *const * __restrict__ sprites,
                                          const T *const * __restrict__ scales,
                                          const T *const * __restrict__ offsets,
                                          const T* __restrict__ backgrounds,
                                          const T* __restrict__ grad_output,

                                          T *const * __restrict__ grad_sprites,
                                          T *const * __restrict__ grad_scales,
                                          T *const * __restrict__ grad_offsets,
                                          T* __restrict__ grad_backgrounds,

                                          const int* __restrict__ pairs,
                                          const int* __restrict__ affecting,
                                          const int* __restrict__ counts,

                                          FlightInfo* __restrict__ flight_info,

                                          const int batch_size,
                                          const int img_height,
                                          const int img_width,
                                          const int n_channels,
                                          const int max_affecting){

  const int output_size = batch_size * img_height * img_width * n_channels;

  const int img_batch_stride = img_height * img_width * n_channels;
  const int img_row_stride = img_width * n_channels;

  const T img_height_T = static_cast<T>(img_height);
  const T img_width_T = static_cast<T>(img_width);

  const T zero = static_cast<T>(0.0);
  const T one = static_cast<T>(1.0);

  CUDA_1D_KERNEL_LOOP(index, output_size) {
    const int pixel_id = index / n_channels;
    const int batch_id = index / img_batch_stride;
    const int index_in_batch = index % img_batch_stride;
    const int img_y = index_in_batch / img_row_stride;
    const int index_in_row = index_in_batch % img_row_stride;
    const int img_x = index_in_row / n_channels;
    const int chan = index_in_row % n_channels;

    const T img_y_T = static_cast<T>(img_y);
    const T img_x_T = static_cast<T>(img_x);

    T weighted_sum = 0.0;
    T importance_sum = 0.0;
    int n_writes = counts[pixel_id];
    T bg = backgrounds[index];
    T _alpha;
    T bg_sum = 0;

    for(int i = 0; i < counts[pixel_id]; ++i) {
      FORWARD_PASS_LOOP_BODY

      _alpha = alpha;
      const T value = alpha * interp + (1-alpha) * bg;
      weighted_sum += importance * value;
      importance_sum += importance;
      bg_sum += importance * (1-alpha);
    }

    T go = grad_output[index];

    if(n_writes == 0){
        grad_backgrounds[index] = go;
    }else if(n_writes == 1){
        grad_backgrounds[index] = go * (1-_alpha);
    }else{
        grad_backgrounds[index] = go * bg_sum / importance_sum;
    }

    for(int i = 0; i < counts[pixel_id]; ++i) {
      FORWARD_PASS_LOOP_BODY

      const T value = alpha * interp + (1-alpha) * bg;

      const T grad_y_wrt_scale_y = -h_T * ((img_y_T + 0.5) / img_height_T - offset_y) / (scale_y * scale_y);
      const T grad_x_wrt_scale_x = -w_T * ((img_x_T + 0.5) / img_width_T - offset_x) / (scale_x * scale_x);

      const T grad_y_wrt_offset_y = -h_T / scale_y;
      const T grad_x_wrt_offset_x = -w_T / scale_x;

      // ------ update gradient through alpha ------

      const T alpha_y_factor = dx * (alpha_fxcy - alpha_fxfy) + (1 - dx) * (alpha_cxcy - alpha_cxfy);
      const T alpha_x_factor = dy * (alpha_cxfy - alpha_fxfy) + (1 - dy) * (alpha_cxcy - alpha_fxcy);

      T alpha_premult = go * (interp - bg);
      if(n_writes > 1){
          alpha_premult *= importance / importance_sum;
      }

      UPDATE_GRAD_SCALES_Y(alpha_premult * alpha_y_factor * grad_y_wrt_scale_y);
      UPDATE_GRAD_SCALES_X(alpha_premult * alpha_x_factor * grad_x_wrt_scale_x);
      UPDATE_GRAD_OFFSETS_Y(alpha_premult * alpha_y_factor * grad_y_wrt_offset_y);
      UPDATE_GRAD_OFFSETS_X(alpha_premult * alpha_x_factor * grad_x_wrt_offset_x);

      if (fx >= 0 && fy >= 0) {
        UPDATE_GRAD_ALPHAS(fx, fy, alpha_premult * dx * dy);
      }
      if (cx <= w - 1 && cy <= h - 1) {
        UPDATE_GRAD_ALPHAS(cx, cy, alpha_premult * (1-dx) * (1-dy));
      }
      if (fx >= 0 && cy <= h - 1) {
        UPDATE_GRAD_ALPHAS(fx, cy, alpha_premult * dx * (1-dy));
      }
      if (cx <= w - 1 && fy >= 0) {
        UPDATE_GRAD_ALPHAS(cx, fy, alpha_premult * (1-dx) * dy);
      }

      // ------ update gradient through importance ------

      const T importance_y_factor = dx * (importance_fxcy - importance_fxfy) + (1 - dx) * (importance_cxcy - importance_cxfy);
      const T importance_x_factor = dy * (importance_cxfy - importance_fxfy) + (1 - dy) * (importance_cxcy - importance_fxcy);

      T importance_premult = 0.0;
      if(n_writes > 1){
          importance_premult = go * (value / importance_sum - weighted_sum / (importance_sum * importance_sum));
      }

      UPDATE_GRAD_SCALES_Y(importance_premult * importance_y_factor * grad_y_wrt_scale_y);
      UPDATE_GRAD_SCALES_X(importance_premult * importance_x_factor * grad_x_wrt_scale_x);
      UPDATE_GRAD_OFFSETS_Y(importance_premult * importance_y_factor * grad_y_wrt_offset_y);
      UPDATE_GRAD_OFFSETS_X(importance_premult * importance_x_factor * grad_x_wrt_offset_x);

      if (fx >= 0 && fy >= 0) {
        UPDATE_GRAD_IMPORTANCES(fx, fy, importance_premult * dx * dy);
      }
      if (cx <= w - 1 && cy <= h - 1) {
        UPDATE_GRAD_IMPORTANCES(cx, cy, importance_premult * (1-dx) * (1-dy));
      }
      if (fx >= 0 && cy <= h - 1) {
        UPDATE_GRAD_IMPORTANCES(fx, cy, importance_premult * dx * (1-dy));
      }
      if (cx <= w - 1 && fy >= 0) {
        UPDATE_GRAD_IMPORTANCES(cx, fy, importance_premult * (1-dx) * dy);
      }

      // ------ update gradient through sprites ------

      T sprite_premult = go * alpha;
      if(n_writes > 1){
          sprite_premult *= importance / importance_sum;
      }

      const T sprite_y_factor = dx * (img_fxcy - img_fxfy) + (1 - dx) * (img_cxcy - img_cxfy);
      const T sprite_x_factor = dy * (img_cxfy - img_fxfy) + (1 - dy) * (img_cxcy - img_fxcy);

      UPDATE_GRAD_SCALES_Y(sprite_premult * sprite_y_factor * grad_y_wrt_scale_y);
      UPDATE_GRAD_SCALES_X(sprite_premult * sprite_x_factor * grad_x_wrt_scale_x);
      UPDATE_GRAD_OFFSETS_Y(sprite_premult * sprite_y_factor * grad_y_wrt_offset_y);
      UPDATE_GRAD_OFFSETS_X(sprite_premult * sprite_x_factor * grad_x_wrt_offset_x);

      if (fx >= 0 && fy >= 0) {
        UPDATE_GRAD_SPRITES(fx, fy, sprite_premult * dx * dy);
      }
      if (cx <= w - 1 && cy <= h - 1) {
        UPDATE_GRAD_SPRITES(cx, cy, sprite_premult * (1-dx) * (1-dy));
      }
      if (fx >= 0 && cy <= h - 1) {
        UPDATE_GRAD_SPRITES(fx, cy, sprite_premult * dx * (1-dy));
      }
      if (cx <= w - 1 && fy >= 0) {
        UPDATE_GRAD_SPRITES(cx, fy, sprite_premult * (1-dx) * dy);
      }
    } // sprite_id - second forward pass
  }
}

#undef FORWARD_PASS_LOOP_BODY
#undef GET_IMPORTANCE_POINT
#undef GET_ALPHA_POINT
#undef GET_SPRITE_POINT
#undef UPDATE_GRAD_SPRITES
#undef UPDATE_GRAD_ALPHAS
#undef UPDATE_GRAD_IMPORTANCES
#undef UPDATE_GRAD_SCALES_Y
#undef UPDATE_GRAD_SCALES_X
#undef UPDATE_GRAD_OFFSETS_Y
#undef UPDATE_GRAD_OFFSETS_X

}  // namespace

namespace functor {

template <typename T>
struct RenderSpritesGrad2DFunctor<GPUDevice, T>{

  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const GPUDevice& d,

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

    // const T *const * sprites_p = &sprites[0];
    // const T *const * scales_p = &scales[0];
    // const T *const * offsets_p = &offsets[0];

    // ::tensorflow::CudaLaunchConfig config;

    // const int n_flights = sprites.size();

    // std::vector<int> n_sprites(n_flights);
    // std::vector<int> sprite_height(n_flights);
    // std::vector<int> sprite_width(n_flights);
    // std::vector<int> sprites_batch_stride(n_flights);
    // std::vector<int> sprites_sprite_stride(n_flights);
    // std::vector<int> sprites_row_stride(n_flights);
    // std::vector<int> scales_batch_stride(n_flights);

    // int n_sprites_per_batch = 0;

    // for(int i=0; i < n_flights; i++){
    //   const int n = shapes[4*i];
    //   const int h = shapes[4*i+1];
    //   const int w = shapes[4*i+2];
    //   const int c = shapes[4*i+3];

    //   n_sprites_per_batch += n;

    //   n_sprites[i] = n;
    //   sprite_height[i] = h;
    //   sprite_width[i] = w;

    //   sprites_batch_stride[i] = n * h * w * c;
    //   sprites_sprite_stride[i] = h * w * c;
    //   sprites_row_stride[i] = w * c;

    //   scales_batch_stride[i] = 2 * n;
    // }

    // int* sprite_height_p = &sprite_height[0];
    // int* sprite_width_p = &sprite_width[0];
    // int* sprites_batch_stride_p = &sprites_batch_stride[0];
    // int* sprites_sprite_stride_p = &sprites_sprite_stride[0];
    // int* sprites_row_stride_p = &sprites_row_stride[0];
    // int* scales_batch_stride_p = &scales_batch_stride[0];

    // std::vector<int> pairs(2*n_sprites_per_batch);
    // int pair_id = 0;
    // for(int i=0; i < n_flights; i++){
    //   for(int j=0; j < n_sprites[i]; j++){
    //     pairs[2*pair_id] = i;
    //     pairs[2*pair_id+1] = j;
    //     pair_id++;
    //   }
    // }
    // int* pairs_p = &pairs[0];

    // const int total_n_sprites = batch_size * n_sprites_per_batch;
    // const int total_n_pixels = batch_size * img_height * img_width;

    // TensorShape counts_shape = {batch_size, img_height, img_width};
    // Tensor counts_tensor;
    // OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, counts_shape, &counts_tensor));

    // int* counts = counts_tensor.flat<int>().data();

    // // --- set counts to zero ---

    // config = ::tensorflow::GetCudaLaunchConfig(total_n_pixels, d);
    // ::tensorflow::SetZero
    //     <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(total_n_pixels, counts);

    // // --- for each pixel, count number of sprites that affect it ---
    // // The -1 indicates that we should just count.

    // config = ::tensorflow::GetCudaLaunchConfig(total_n_sprites, d);
    // // index_affecting_sprites<T>
    // //     <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
    // //         scales_p, offsets_p, pairs_p, nullptr, counts,
    // //         scales_batch_stride_p, sprite_height_p, sprite_width_p,
    // //         total_n_sprites, n_sprites_per_batch, img_height, img_width, -1);

    // // --- get max of counts so we know how much space to allocate for array `affecting` ---

    // const int maxBlocks = 64;
    // const int maxThreads = 256;

    // TensorShape temp_shape = {maxBlocks};
    // Tensor temp_tensor;
    // OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, temp_shape, &temp_tensor));
    // int* temp = temp_tensor.flat<int>().data();

    // const int max_affecting = reduce_max(d, total_n_pixels, maxThreads, maxBlocks, counts, temp);

    // // --- set counts to zero again ---

    // config = ::tensorflow::GetCudaLaunchConfig(total_n_pixels, d);
    // ::tensorflow::SetZero
    //     <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(total_n_pixels, counts);

    // // --- for each pixel, find the indices of the sprites that affect it ---

    // TensorShape affecting_shape = {batch_size, img_height, img_width, max_affecting};
    // Tensor affecting_tensor;
    // OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, affecting_shape, &affecting_tensor));

    // int* affecting = affecting_tensor.flat<int>().data();

    // config = ::tensorflow::GetCudaLaunchConfig(total_n_sprites, d);
    // // index_affecting_sprites<T>
    // //     <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
    // //         scales_p, offsets_p, pairs_p, affecting, counts,
    // //         scales_batch_stride_p, sprite_height_p, sprite_width_p,
    // //         total_n_sprites, n_sprites_per_batch, img_height, img_width, max_affecting);

    FUNCTOR_SETUP

    // --- set gradients to 0 ---
    // (because the kernel incrementally updates the tensor entries by adding partial contributions,
    //  except for grad_backgrounds, which is only set once).

    for(int flight_id=0; flight_id < n_flights; flight_id++) {
      const int n = flight_info[flight_id].n_sprites;
      const int h = flight_info[flight_id].sprite_height;
      const int w = flight_info[flight_id].sprite_width;
      const int grad_sprites_size = batch_size * n * h * w * (n_channels + 2);
      const int grad_scales_size = batch_size * n * 2;

      config = ::tensorflow::GetCudaLaunchConfig(grad_sprites_size, d);
      ::tensorflow::SetZero
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_sprites_size, grad_sprites[flight_id]);

      config = ::tensorflow::GetCudaLaunchConfig(grad_scales_size, d);
      ::tensorflow::SetZero
            <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_scales_size, grad_scales[flight_id]);

      config = ::tensorflow::GetCudaLaunchConfig(grad_scales_size, d);
      ::tensorflow::SetZero
          <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(grad_scales_size, grad_offsets[flight_id]);
    }

    // copy grad pointer arrays to device

    Tensor grad_sprites_t;
    T** grad_sprites_d;
    allocate_temp_array<T*>(ctx, &grad_sprites_d, grad_sprites.size(), &grad_sprites_t);
    cudaMemcpy(grad_sprites_d, &grad_sprites[0], sizeof(T*)*n_flights, cudaMemcpyHostToDevice);

    Tensor grad_scales_t;
    T** grad_scales_d;
    allocate_temp_array<T*>(ctx, &grad_scales_d, grad_scales.size(), &grad_scales_t);
    cudaMemcpy(grad_scales_d, &grad_scales[0], sizeof(T*)*n_flights, cudaMemcpyHostToDevice);

    Tensor grad_offsets_t;
    T** grad_offsets_d;
    allocate_temp_array<T*>(ctx, &grad_offsets_d, grad_offsets.size(), &grad_offsets_t);
    cudaMemcpy(grad_offsets_d, &grad_offsets[0], sizeof(T*)*n_flights, cudaMemcpyHostToDevice);

    // --- compute the image ---
    const int output_size = batch_size * img_height * img_width * n_channels;
    config = ::tensorflow::GetCudaLaunchConfig(output_size, d, RenderSpritesGrad2DKernel<T>, 0, 0);
    RenderSpritesGrad2DKernel<T>
        <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
            sprites_d, scales_d, offsets_d, backgrounds, grad_output,
            grad_sprites_d, grad_scales_d, grad_offsets_d, grad_backgrounds,
            pairs_d, affecting_d, counts_d, flight_info_d,
            batch_size, img_height, img_width, n_channels, max_affecting);
  }
};

template struct RenderSpritesGrad2DFunctor<GPUDevice, float>;

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
