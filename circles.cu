#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SQRT(x) (sqrtf(x))

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void render_cpu(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue) {

    // Initialize background to white
    for (int32_t pixel_idx = 0; pixel_idx < width * height; pixel_idx++) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }

    // Render circles
    for (int32_t i = 0; i < n_circle; i++) {
        float c_x = circle_x[i];
        float c_y = circle_y[i];
        float c_radius = circle_radius[i];
        for (int32_t y = int32_t(c_y - c_radius); y <= int32_t(c_y + c_radius + 1.0f);
             y++) {
            for (int32_t x = int32_t(c_x - c_radius); x <= int32_t(c_x + c_radius + 1.0f);
                 x++) {
                float dx = x - c_x;
                float dy = y - c_y;
                if (!(0 <= x && x < width && 0 <= y && y < height &&
                      dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }
                int32_t pixel_idx = y * width + x;
                float pixel_red = img_red[pixel_idx];
                float pixel_green = img_green[pixel_idx];
                float pixel_blue = img_blue[pixel_idx];
                float pixel_alpha = circle_alpha[i];
                pixel_red =
                    circle_red[i] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                pixel_green =
                    circle_green[i] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
                pixel_blue =
                    circle_blue[i] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
                img_red[pixel_idx] = pixel_red;
                img_green[pixel_idx] = pixel_green;
                img_blue[pixel_idx] = pixel_blue;
            }
        }
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

#define TILE_WIDTH 128
#define TILE_HEIGHT 128

#define THREADS_PER_WARP 32
#define WARPS_PER_BLOCK 32
#define CIRCLES_PER_THREAD 4
#define CIRCLES_PER_BLOCK 1024
#define THREADS_PER_BLOCK 1024 // one thread per pixel
#define SCAN_VALS_THREAD 4     // TODO: maybe lift this for organize_circles kernel too
#define SPINE_VALS_THREAD 8
#define CIRCLE_BATCH_SIZE 10000

#define PIXELS_PER_THREAD TILE_WIDTH *TILE_HEIGHT / THREADS_PER_BLOCK
#define THREAD_TILE_DIM 4 // hardcoded

namespace circles_gpu {

__global__ void compute_tile_mask(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    int32_t num_tiles_x,
    int32_t num_tiles_y,
    float const *circle_x, // pointer to GPU memory
    float const *circle_y, // pointer to GPU memory
    float const *circle_radius,
    int8_t *mask) {

    int32_t circle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (circle_idx >= n_circle) {
        return;
    }
    float c_x = circle_x[circle_idx];
    float c_y = circle_y[circle_idx];
    float c_radius = circle_radius[circle_idx];

    // get bbox coordinates in tile space
    int box_x_min = max(int((c_x - c_radius) / TILE_WIDTH), 0);
    int box_x_max = min(int((c_x + c_radius) / TILE_WIDTH), num_tiles_x - 1);
    int box_y_min = max(int((c_y - c_radius) / TILE_HEIGHT), 0);
    int box_y_max = min(int((c_y + c_radius) / TILE_HEIGHT), num_tiles_y - 1);

    for (int box_y = box_y_min; box_y <= box_y_max; box_y++) {
        for (int box_x = box_x_min; box_x <= box_x_max; box_x++) {
            int tile_id = box_y * num_tiles_x + box_x;
            size_t i = (size_t)tile_id * n_circle + circle_idx;
            mask[i] = 1;
        }
    }
}

__global__ void upstream_reduction(
    int32_t n_circles,
    int8_t *mask, // pointer to GPU memory
    uint32_t *blocksums) {

    extern __shared__ __align__(16) uint32_t shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    int threadId = threadIdx.x;
    int threads_per_block = blockDim.x;
    int lane = threadId % THREADS_PER_WARP;
    int warpId = threadId / THREADS_PER_WARP;
    int block_offset = blockIdx.x * threads_per_block * SCAN_VALS_THREAD;

    // reduce within thread
    uint32_t vals[SCAN_VALS_THREAD];
    uint32_t thread_total = 0;
#pragma unroll
    for (int i = 0; i < SCAN_VALS_THREAD; i++) {
        int idx = block_offset + threadId * SCAN_VALS_THREAD + i;
        vals[i] = (idx < n_circles) ? mask[idx] : 0;
        thread_total += vals[i];
    }
    // reduce within warp
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_total += __shfl_down_sync(0xffffffff, thread_total, offset);
    }

    // Write warp result
    if (lane == 0) {
        shmem[warpId] = thread_total;
    }
    __syncthreads();

    // reduce across warps
    if (warpId == 0) {
        uint32_t warp_sum = shmem[lane];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }

        if (lane == 0)
            blocksums[blockIdx.x] = warp_sum;
    }
}

__global__ void spine_scan(
    int32_t num_blocks,
    int32_t tile_id,
    uint32_t *blocksums, // pointer to GPU memory
    uint32_t *tile_to_circle_count) {
    extern __shared__ __align__(16) uint32_t shmem_raw[]; // OK
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    // int threads_per_block = blockDim.x;
    int threadId = threadIdx.x;
    int lane = threadId % THREADS_PER_WARP;
    int warpId = threadId / THREADS_PER_WARP;

    // scan within thread
    uint32_t vals[SPINE_VALS_THREAD];
    for (int i = 0; i < SPINE_VALS_THREAD; i++) {
        int idx = threadId * SPINE_VALS_THREAD + i;
        vals[i] = blocksums[idx];
    }
    for (int i = 1; i < SPINE_VALS_THREAD; i++) {
        vals[i] = vals[i - 1] + vals[i];
    }

    // scan within warp
    uint32_t thread_total = vals[SPINE_VALS_THREAD - 1];
    uint32_t warp_prefix = thread_total;
    for (int offset = 1; offset < THREADS_PER_WARP; offset <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, warp_prefix, offset);
        if (lane >= offset)
            warp_prefix += n;
    }
    if (lane == THREADS_PER_WARP - 1) {
        shmem[warpId] = warp_prefix; // last lane holds warp sum
    }
    __syncthreads();

    // scan across warps
    if (warpId == 0) {
        uint32_t warp_sum = (threadId < THREADS_PER_WARP) ? shmem[lane] : 0;
        for (int offset = 1; offset < THREADS_PER_WARP; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (lane >= offset)
                warp_sum += n;
        }
        if (threadId < 32)
            shmem[lane] = warp_sum;
    }
    __syncthreads();

    if (warpId > 0) {
        warp_prefix += shmem[warpId - 1];
    }

    for (int i = 0; i < SPINE_VALS_THREAD; i++) {
        vals[i] += (warp_prefix - thread_total); // warp_prefix includes this thread_total
    }

    for (int i = 0; i < SPINE_VALS_THREAD; i++) {
        int idx = threadId * SPINE_VALS_THREAD + i;
        blocksums[idx] = vals[i];
    }
    __syncthreads();

    // first thread writes total count
    if (threadId == 0) {
        tile_to_circle_count[tile_id] = blocksums[num_blocks - 1];
    }
}

__global__ void downstream_scan(
    int32_t n_circle,
    int32_t num_blocks,
    int32_t tile_id,
    int8_t *mask,                  // pointer to GPU memory
    uint32_t *blocksums,           // pointer to GPU memory
    uint32_t *tile_to_circle_list) // pointer to GPU memory
{
    extern __shared__ __align__(16) uint32_t shmem_raw[]; // OK
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * SCAN_VALS_THREAD;
    int threadId = threadIdx.x;
    int lane = threadId % THREADS_PER_WARP;
    int warpId = threadId / THREADS_PER_WARP;

    // load from global memory & perform thread scan
    uint32_t vals[SCAN_VALS_THREAD];
    for (int i = 0; i < SCAN_VALS_THREAD; i++) {
        int idx = block_offset + threadId * SCAN_VALS_THREAD + i;
        vals[i] = mask[idx];
    }
    for (int i = 1; i < SCAN_VALS_THREAD; i++) {
        vals[i] = vals[i - 1] + vals[i];
    }

    // scan within warp
    uint32_t thread_total = vals[SCAN_VALS_THREAD - 1];
    uint32_t warp_prefix = thread_total;
    for (int offset = 1; offset < THREADS_PER_WARP; offset <<= 1) {
        uint32_t n = __shfl_up_sync(0xFFFFFFFF, warp_prefix, offset);
        if (lane >= offset)
            warp_prefix += n;
    }
    if (lane == THREADS_PER_WARP - 1) {
        shmem[warpId] = warp_prefix; // last lane holds warp sum
    }
    __syncthreads();

    // scan across warps
    if (warpId == 0) {
        uint32_t warp_sum = (lane < WARPS_PER_BLOCK) ? shmem[lane] : 0;
        for (int offset = 1; offset < WARPS_PER_BLOCK; offset <<= 1) {
            uint32_t n = __shfl_up_sync(0xFFFFFFFF, warp_sum, offset);
            if (lane >= offset)
                warp_sum += n;
        }
        if (threadId < 32)
            shmem[lane] = warp_sum;
    }
    __syncthreads();

    if (warpId > 0) {
        warp_prefix += shmem[warpId - 1];
    }
    for (int i = 0; i < SCAN_VALS_THREAD; i++) {
        vals[i] += (warp_prefix - thread_total); // warp_prefix includes this thread_total
    }

    // add block prefix
    uint32_t block_prefix = 0;
    if (blockIdx.x > 0) {
        block_prefix = blocksums[blockIdx.x - 1];
    }
    for (int i = 0; i < SCAN_VALS_THREAD; i++) {
        // int idx = block_offset + threadId * SCAN_VALS_THREAD + i;
        vals[i] = block_prefix + vals[i];
    }

    for (int i = 0; i < SCAN_VALS_THREAD; i++) {
        int idx = block_offset + threadId * SCAN_VALS_THREAD + i;
        if (idx >= n_circle) {
            continue;
        }
        if (mask[idx]) {
            int write_idx = vals[i] - 1;
            tile_to_circle_list[write_idx] = idx % n_circle;
        }
    }
    __syncthreads();
    // first thread writes total count
    // if (blockIdx.x == 0 && threadId == 0) {
    //     tile_to_circle_count[tile_id] = blocksums[num_blocks - 1];
    // }
}

__global__ void render_kernel(
    int32_t n_circle,
    int32_t width,
    int32_t height,
    int32_t num_tiles_x,
    int32_t num_tiles_y,
    uint32_t *circle_list_all,
    uint32_t *circle_count,
    float const *circle_x,      // pointer to GPU memory
    float const *circle_y,      // pointer to GPU memory
    float const *circle_radius, // pointer to GPU memory
    float const *circle_red,    // pointer to GPU memory
    float const *circle_green,  // pointer to GPU memory
    float const *circle_blue,   // pointer to GPU memory
    float const *circle_alpha,  // pointer to GPU memory
    float *img_red,             // pointer to GPU memory
    float *img_green,           // pointer to GPU memory
    float *img_blue             // pointer to GPU memory
) {
    extern __shared__ __align__(16) float shmem[];
    float *shircle_x = shmem;
    float *shircle_y = shircle_x + THREADS_PER_BLOCK;
    float *shircle_radius = shircle_y + THREADS_PER_BLOCK;
    float *shircle_red = shircle_radius + THREADS_PER_BLOCK;
    float *shircle_green = shircle_red + THREADS_PER_BLOCK;
    float *shircle_blue = shircle_green + THREADS_PER_BLOCK;
    float *shircle_alpha = shircle_blue + THREADS_PER_BLOCK;

    // per-thread pixel rgb vals
    // int num_tiles = num_tiles_x * num_tiles_y;
    // float *thread_pixel_red = shircle_alpha + THREADS_PER_BLOCK;
    // float *thread_pixel_green =
    //     thread_pixel_red + THREAD_TILE_DIM * THREAD_TILE_DIM * THREADS_PER_BLOCK;
    // float *thread_pixel_blue =
    //     thread_pixel_green + THREAD_TILE_DIM * THREAD_TILE_DIM * THREADS_PER_BLOCK;

    int num_circles = circle_count[blockIdx.x]; // number of circles affecting this tile

    int threadId = threadIdx.x;
    int threads_per_block = blockDim.x;

    int tile_x = blockIdx.x % num_tiles_x;
    int tile_y = blockIdx.x / num_tiles_x;
    int tile_id = tile_y * num_tiles_x + tile_x;
    uint32_t *circle_list =
        circle_list_all + tile_id * n_circle; // circle list for this tile

    int t_x = threadId % (int)(TILE_WIDTH / THREAD_TILE_DIM);
    int t_y = threadId / (int)(TILE_WIDTH / THREAD_TILE_DIM);

    int base_x = tile_x * TILE_WIDTH + t_x * THREAD_TILE_DIM;
    int base_y = tile_y * TILE_HEIGHT + t_y * THREAD_TILE_DIM;

    // initialize to white
#pragma unroll
    for (int y = 0; y < THREAD_TILE_DIM; y++) {
        int pixel_y = base_y + y;

        float4 white_vec = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        reinterpret_cast<float4 *>(img_red)[(pixel_y * width + base_x) / 4] = white_vec;
        reinterpret_cast<float4 *>(img_green)[(pixel_y * width + base_x) / 4] = white_vec;
        reinterpret_cast<float4 *>(img_blue)[(pixel_y * width + base_x) / 4] = white_vec;
    }

    float img_red_val[4] = {0};
    float img_green_val[4] = {0};
    float img_blue_val[4] = {0};

#pragma unroll
    for (int i = 0; i < 4; i++) {
        img_red_val[i] = 1.0f;
        img_green_val[i] = 1.0f;
        img_blue_val[i] = 1.0f;
    }
    for (int32_t i = 0; i < num_circles; i += THREADS_PER_BLOCK) {
        // each thread loads one circle
        int circle_idx = i + threadId;
        if (circle_idx < num_circles) {
            int c_idx = circle_list[circle_idx];
            shircle_x[threadId] = circle_x[c_idx];
            shircle_y[threadId] = circle_y[c_idx];
            shircle_radius[threadId] = circle_radius[c_idx];
            shircle_red[threadId] = circle_red[c_idx];
            shircle_green[threadId] = circle_green[c_idx];
            shircle_blue[threadId] = circle_blue[c_idx];
            shircle_alpha[threadId] = circle_alpha[c_idx];
        }
        __syncthreads();

        int circles_in_batch = min(THREADS_PER_BLOCK, num_circles - i);
        for (int j = 0; j < circles_in_batch; j++) {
            int c_idx = j;

            // int c_idx = circle_list[i];
            float c_x = shircle_x[c_idx];
            float c_y = shircle_y[c_idx];
            float c_radius = shircle_radius[c_idx];
            float pixel_alpha = shircle_alpha[c_idx];

            // process the pixels with space in registers (4)
            for (int x = 0; x < 4; x++) {
                int y = 0;
                int pixel_x = base_x + x;
                int pixel_y = base_y + y;

                float pixel_red = img_red_val[x];
                float pixel_green = img_green_val[x];
                float pixel_blue = img_blue_val[x];
                // // check if pixel is in circle
                float dx = pixel_x - c_x;
                float dy = pixel_y - c_y;

                if (!(dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }

                img_red_val[x] =
                    shircle_red[c_idx] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                img_green_val[x] = shircle_green[c_idx] * pixel_alpha +
                    pixel_green * (1.0f - pixel_alpha);
                img_blue_val[x] =
                    shircle_blue[c_idx] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
            }
            for (int x = 0; x < THREAD_TILE_DIM; x++) {
                for (int y = 1; y < THREAD_TILE_DIM; y++) {

                    int pixel_x = base_x + x;
                    int pixel_y = base_y + y;
                    if (pixel_y >= height) { // this check does nothing but idk
                                             // keeping it is faster..
                        break;
                    }

                    int pixel_idx = pixel_y * width + pixel_x;

                    float pixel_red = img_red[pixel_idx];
                    float pixel_green =
                        img_green[pixel_idx] == 0 ? 1.0f : img_green[pixel_idx];
                    float pixel_blue = img_blue[pixel_idx];
                    // // check if pixel is in circle
                    float dx = pixel_x - c_x;
                    float dy = pixel_y - c_y;

                    if (!(dx * dx + dy * dy < c_radius * c_radius)) {
                        continue;
                    }

                    pixel_red = shircle_red[c_idx] * pixel_alpha +
                        pixel_red * (1.0f - pixel_alpha);
                    pixel_green = shircle_green[c_idx] * pixel_alpha +
                        pixel_green * (1.0f - pixel_alpha);
                    pixel_blue = shircle_blue[c_idx] * pixel_alpha +
                        pixel_blue * (1.0f - pixel_alpha);

                    img_red[pixel_idx] = pixel_red;
                    // img_red_val[y * THREAD_TILE_DIM + x] = pixel_red;
                    img_green[pixel_idx] = pixel_green;
                    img_blue[pixel_idx] = pixel_blue;
                }
            }
        }
    }
    // write back per-thread pixel rgb vals
#pragma unroll
    for (int x = 0; x < 4; x++) {
        int pixel_y = base_y;
        int pixel_x = base_x + x;
        int pixel_idx = pixel_y * width + pixel_x;
        img_red[pixel_idx] = img_red_val[x];
        img_green[pixel_idx] = img_green_val[x];
        img_blue[pixel_idx] = img_blue_val[x];
    }
}

void launch_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,      // pointer to GPU memory
    float const *circle_y,      // pointer to GPU memory
    float const *circle_radius, // pointer to GPU memory
    float const *circle_red,    // pointer to GPU memory
    float const *circle_green,  // pointer to GPU memory
    float const *circle_blue,   // pointer to GPU memory
    float const *circle_alpha,  // pointer to GPU memory
    float *img_red,             // pointer to GPU memory
    float *img_green,           // pointer to GPU memory
    float *img_blue,            // pointer to GPU memory
    GpuMemoryPool &memory_pool) {

    int32_t num_tiles_x = CEIL_DIV(width, TILE_WIDTH);
    int32_t num_tiles_y = CEIL_DIV(height, TILE_HEIGHT);
    int32_t num_tiles = num_tiles_x * num_tiles_y;

    size_t mask_size = num_tiles * n_circle * sizeof(int8_t);
    int8_t *mask = reinterpret_cast<int8_t *>(memory_pool.alloc(mask_size));

    size_t tile_to_circle_list_size = num_tiles * n_circle * sizeof(uint32_t);
    uint32_t *tile_to_circle_list =
        reinterpret_cast<uint32_t *>(memory_pool.alloc(tile_to_circle_list_size));

    // number of circles per tile
    size_t tile_circle_count_size = num_tiles * sizeof(uint32_t);
    uint32_t *tile_circle_count =
        reinterpret_cast<uint32_t *>(memory_pool.alloc(tile_circle_count_size));

    dim3 gridDim(CEIL_DIV(n_circle, CIRCLES_PER_BLOCK));
    dim3 blockDim(CIRCLES_PER_BLOCK);

    compute_tile_mask<<<gridDim, blockDim>>>(
        width,
        height,
        n_circle,
        num_tiles_x,
        num_tiles_y,
        circle_x,
        circle_y,
        circle_radius,
        mask);

    int32_t num_blocks_scan = CEIL_DIV(n_circle, THREADS_PER_BLOCK * SCAN_VALS_THREAD);
    dim3 upstreamGridDim(num_blocks_scan);
    dim3 upstreamBlockDim(THREADS_PER_BLOCK);
    size_t upstream_shmem_size_bytes = THREADS_PER_BLOCK * sizeof(uint32_t);

    // temp workspace for upstream scan
    size_t upstream_workspace_size = upstreamGridDim.x * sizeof(uint32_t);
    uint32_t *upstream_workspace =
        reinterpret_cast<uint32_t *>(memory_pool.alloc(upstream_workspace_size));

    dim3 spineGridDim(CEIL_DIV(upstreamGridDim.x, THREADS_PER_BLOCK));
    dim3 spineBlockDim(THREADS_PER_BLOCK);

    for (int32_t tile_x = 0; tile_x < num_tiles_x; tile_x++) {
        for (int32_t tile_y = 0; tile_y < num_tiles_y; tile_y++) {
            // perform scan on mask to get circle list for this tile
            int32_t tile_id = tile_y * num_tiles_x + tile_x;

            upstream_reduction<<<
                upstreamGridDim,
                upstreamBlockDim,
                upstream_shmem_size_bytes>>>(
                n_circle,
                mask + tile_id * n_circle,
                upstream_workspace);

            // spine scan
            spine_scan<<<spineGridDim, spineBlockDim, upstream_shmem_size_bytes>>>(
                upstreamGridDim.x,
                tile_id,
                upstream_workspace,
                tile_circle_count);

            // downstream scan
            downstream_scan<<<
                upstreamGridDim,
                upstreamBlockDim,
                upstream_shmem_size_bytes>>>(
                n_circle,
                upstreamGridDim.x,
                tile_id,
                mask + tile_id * n_circle,
                upstream_workspace,
                tile_to_circle_list + tile_id * n_circle);
        }
    }
    // render tiles
    dim3 renderGridDim(num_tiles);
    dim3 renderBlockDim(THREADS_PER_BLOCK);
    // shmem stores THREADS_PER_BLOCK circles + each thread's rgb vals
    size_t shmem_size_bytes = THREADS_PER_BLOCK * 7 * sizeof(float);

    // +TILE_HEIGHT * TILE_WIDTH * 3 * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(
        render_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size_bytes));

    render_kernel<<<renderGridDim, renderBlockDim, shmem_size_bytes>>>(
        n_circle,
        width,
        height,
        num_tiles_x,
        num_tiles_y,
        tile_to_circle_list,
        tile_circle_count,
        circle_x,
        circle_y,
        circle_radius,
        circle_red,
        circle_green,
        circle_blue,
        circle_alpha,
        img_red,
        img_green,
        img_blue);

    // dim3 gridDim2(num_tiles);
    // dim3 blockDim2(THREADS_PER_BLOCK);
    // size_t shmem_size_bytes =
    //     THREADS_PER_BLOCK * SPINE_VALS_THREAD * sizeof(uint32_t) +
    //     MIN(n_circle, CIRCLE_BATCH_SIZE) * sizeof(uint32_t);
    // CUDA_CHECK(cudaFuncSetAttribute(
    //     tile_scan_and_render,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //     shmem_size_bytes));
    // tile_scan_and_render<<<gridDim2, blockDim2, shmem_size_bytes>>>(
    //     width,
    //     height,
    //     batch_size,
    //     mask,
    //     circle_x + circle_start,
    //     circle_y + circle_start,
    //     circle_radius + circle_start,
    //     circle_red + circle_start,
    //     circle_green + circle_start,
    //     circle_blue + circle_start,
    //     circle_alpha + circle_start,
    //     img_red,
    //     img_green,
    //     img_blue);
}

} // namespace circles_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Scene {
    int32_t width;
    int32_t height;
    std::vector<float> circle_x;
    std::vector<float> circle_y;
    std::vector<float> circle_radius;
    std::vector<float> circle_red;
    std::vector<float> circle_green;
    std::vector<float> circle_blue;
    std::vector<float> circle_alpha;

    int32_t n_circle() const { return circle_x.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

float max_abs_diff(Image const &a, Image const &b) {
    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_red = std::abs(a.red.at(idx) - b.red.at(idx));
        float diff_green = std::abs(a.green.at(idx) - b.green.at(idx));
        float diff_blue = std::abs(a.blue.at(idx) - b.blue.at(idx));
        max_diff = std::max(max_diff, diff_red);
        max_diff = std::max(max_diff, diff_green);
        max_diff = std::max(max_diff, diff_blue);
    }
    return max_diff;
}

struct Results {
    bool correct;
    float max_abs_diff;
    Image image_expected;
    Image image_actual;
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    render_cpu(
        scene.width,
        scene.height,
        scene.n_circle(),
        scene.circle_x.data(),
        scene.circle_y.data(),
        scene.circle_radius.data(),
        scene.circle_red.data(),
        scene.circle_green.data(),
        scene.circle_blue.data(),
        scene.circle_alpha.data(),
        img_expected.red.data(),
        img_expected.green.data(),
        img_expected.blue.data());

    auto circle_x_gpu = GpuBuf<float>(scene.circle_x);
    auto circle_y_gpu = GpuBuf<float>(scene.circle_y);
    auto circle_radius_gpu = GpuBuf<float>(scene.circle_radius);
    auto circle_red_gpu = GpuBuf<float>(scene.circle_red);
    auto circle_green_gpu = GpuBuf<float>(scene.circle_green);
    auto circle_blue_gpu = GpuBuf<float>(scene.circle_blue);
    auto circle_alpha_gpu = GpuBuf<float>(scene.circle_alpha);
    auto img_red_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_green_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_blue_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(img_red_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            img_green_gpu.data,
            0,
            scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(img_blue_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        circles_gpu::launch_render(
            scene.width,
            scene.height,
            scene.n_circle(),
            circle_x_gpu.data,
            circle_y_gpu.data,
            circle_radius_gpu.data,
            circle_red_gpu.data,
            circle_green_gpu.data,
            circle_blue_gpu.data,
            circle_alpha_gpu.data,
            img_red_gpu.data,
            img_green_gpu.data,
            img_blue_gpu.data,
            memory_pool);
    };

    reset();
    f();

    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    CUDA_CHECK(cudaMemcpy(
        img_actual.red.data(),
        img_red_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.green.data(),
        img_green_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.blue.data(),
        img_blue_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    if (max_diff > 5e-2) {
        return Results{
            false,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    if (mode == Mode::TEST) {
        return Results{
            true,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

template <typename Rng>
Scene gen_random(Rng &rng, int32_t width, int32_t height, int32_t n_circle) {
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto z_values = std::vector<float>();
    for (int32_t i = 0; i < n_circle; i++) {
        float z;
        for (;;) {
            z = unif_0_1(rng);
            z = std::max(z, unif_0_1(rng));
            if (z > 0.01) {
                break;
            }
        }
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        z_values.push_back(z);
    }
    std::sort(z_values.begin(), z_values.end(), std::greater<float>());

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };
    auto color_idx_dist = std::uniform_int_distribution<int>(0, colors.size() - 1);
    auto alpha_dist = std::uniform_real_distribution<float>(0.0f, 0.3f);

    int32_t fog_interval = n_circle / 10;
    float fog_alpha = 0.2;

    auto scene = Scene{width, height};
    float base_radius_scale = 1.0f;
    int32_t i = 0;
    for (float z : z_values) {
        float max_radius = base_radius_scale / z;
        float radius = std::max(1.0f, unif_0_1(rng) * max_radius);
        float x = unif_0_1(rng) * (width + 2 * max_radius) - max_radius;
        float y = unif_0_1(rng) * (height + 2 * max_radius) - max_radius;
        int color_idx = color_idx_dist(rng);
        uint32_t color = colors[color_idx];
        scene.circle_x.push_back(x);
        scene.circle_y.push_back(y);
        scene.circle_radius.push_back(radius);
        scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
        scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
        scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
        scene.circle_alpha.push_back(alpha_dist(rng));
        i++;
        if (i % fog_interval == 0 && i + 1 < n_circle) {
            scene.circle_x.push_back(float(width - 1) / 2.0f);
            scene.circle_y.push_back(float(height - 1) / 2.0f);
            scene.circle_radius.push_back(float(std::max(width, height)));
            scene.circle_red.push_back(1.0f);
            scene.circle_green.push_back(1.0f);
            scene.circle_blue.push_back(1.0f);
            scene.circle_alpha.push_back(fog_alpha);
        }
    }

    return scene;
}

constexpr float PI = 3.14159265359f;

Scene gen_overlapping_opaque() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };

    int32_t n_circle = 20;
    int32_t n_ring = 4;
    float angle_range = PI;
    for (int32_t ring = 0; ring < n_ring; ring++) {
        float dist = 20.0f * (ring + 1);
        float saturation = float(ring + 1) / n_ring;
        float hue_shift = float(ring) / (n_ring - 1);
        for (int32_t i = 0; i < n_circle; i++) {
            float theta = angle_range * i / (n_circle - 1);
            float x = width / 2.0f - dist * std::cos(theta);
            float y = height / 2.0f - dist * std::sin(theta);
            scene.circle_x.push_back(x);
            scene.circle_y.push_back(y);
            scene.circle_radius.push_back(16.0f);
            auto color = colors[(i + ring * 2) % colors.size()];
            scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
            scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
            scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
            scene.circle_alpha.push_back(1.0f);
        }
    }

    return scene;
}

Scene gen_overlapping_transparent() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    float offset = 20.0f;
    float radius = 40.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}

Scene gen_simple() {
    /*
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    */
    int32_t width = 256;
    int32_t height = 256;
    auto scene = Scene{width, height};
    scene.circle_x = std::vector<float>{
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
    };
    scene.circle_radius = std::vector<float>{
        40.0f,
        40.0f,
        40.0f,
        40.0f,
    };
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0xcc) / 255.0f,
        float(0x20) / 255.0f,
        float(0x28) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x9f) / 255.0f,
        float(0x80) / 255.0f,
        float(0x74) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0x26) / 255.0f,
        float(0x20) / 255.0f,
        float(0xaa) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };
    return scene;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void write_bmp(
    std::string const &fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
}

uint8_t float_to_byte(float x) {
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_image(std::string const &fname, Image const &img) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        float red = img.red.at(idx);
        float green = img.green.at(idx);
        float blue = img.blue.at(idx);
        // BMP stores pixels in BGR order
        pixels.at(idx * 3) = float_to_byte(blue);
        pixels.at(idx * 3 + 1) = float_to_byte(green);
        pixels.at(idx * 3 + 2) = float_to_byte(red);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

Image compute_img_diff(Image const &a, Image const &b) {
    auto img_diff = Image{
        a.width,
        a.height,
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
    };
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        img_diff.red.at(idx) = std::abs(a.red.at(idx) - b.red.at(idx));
        img_diff.green.at(idx) = std::abs(a.green.at(idx) - b.green.at(idx));
        img_diff.blue.at(idx) = std::abs(a.blue.at(idx) - b.blue.at(idx));
    }
    return img_diff;
}

struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto scenes = std::vector<SceneTest>();
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
    scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    scenes.push_back(
        {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    scenes.push_back(
        {"ten_million_circles",
         Mode::BENCHMARK,
         gen_random(rng, 1024, 1024, 10'000'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_cpu.bmp",
            results.image_expected);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_gpu.bmp",
            results.image_actual);
        if (!results.correct) {
            printf("  Result did not match expected image\n");
            printf("  Max absolute difference: %.2e\n", results.max_abs_diff);
            auto diff = compute_img_diff(results.image_expected, results.image_actual);
            write_image(
                std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                    "_diff.bmp",
                diff);
            printf(
                "  (Wrote image diff to 'out/img%d_%s_diff.bmp')\n",
                i,
                scene_test.name.c_str());
            fail_count++;
            continue;
        } else {
            printf("  OK\n");
        }
        if (scene_test.mode == Mode::BENCHMARK) {
            printf("  Time: %f ms\n", results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
