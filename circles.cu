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

#define TILE_WIDTH 32
#define TILE_HEIGHT 32

#define CIRCLES_PER_BLOCK 256
#define THREADS_PER_BLOCK 1024    // one thread per pixel
#define CIRCLES_PER_THREAD_SCAN 4 // TODO: maybe lift this for organize_circles kernel too
#define SPINE_VALS_PER_THREAD 4
#define CIRCLE_BATCH_SIZE 4096

namespace circles_gpu {

__global__ void render_gpu_naive(
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
    int box_x_max = min(int((c_x + c_radius) / TILE_WIDTH), (width - 1) / TILE_WIDTH);
    int box_y_min = max(int((c_y - c_radius) / TILE_HEIGHT), 0);
    int box_y_max = min(int((c_y + c_radius) / TILE_HEIGHT), (height - 1) / TILE_HEIGHT);

    for (int box_y = box_y_min; box_y <= box_y_max; box_y++) {
        for (int box_x = box_x_min; box_x <= box_x_max; box_x++) {
            int tile_id = box_y * num_tiles_x + box_x;
            mask[tile_id * n_circle + circle_idx] = 1;
        }
    }
}

__global__ void tile_scan_and_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    int8_t *mask,               // pointer to GPU memory
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
    extern __shared__ __align__(16) uint32_t shmem_raw[]; // OK
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);
    uint32_t *circle_list = shmem + THREADS_PER_BLOCK;

    int threads_per_block = blockDim.x;
    int threadId = threadIdx.x;
    int mask_offset = blockIdx.x * n_circle;

    uint32_t vals[SPINE_VALS_PER_THREAD];
    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        int idx = mask_offset + threadId * SPINE_VALS_PER_THREAD + i;
        if (threadId * SPINE_VALS_PER_THREAD + i < n_circle) {
            vals[i] = mask[idx];
        } else {
            vals[i] = 0;
        }
    }
    for (int i = 1; i < SPINE_VALS_PER_THREAD; i++) {
        vals[i] = vals[i - 1] + vals[i];
    }

    uint32_t thread_sum = vals[SPINE_VALS_PER_THREAD - 1];
    shmem[threadId] = thread_sum;

    // scan across shmem
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        uint32_t cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = shmem[threadId - i] + cur_val;
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();
    int num_block_circles = shmem[threads_per_block - 1];

    uint32_t threadPrefix = 0;
    if (threadId > 0) {
        threadPrefix = shmem[threadId - 1];
    }
    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        vals[i] = threadPrefix + vals[i];
    }

    // write circle list
    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        int idx = mask_offset + threadId * SPINE_VALS_PER_THREAD + i;
        if (threadId * SPINE_VALS_PER_THREAD + i >= n_circle) {
            continue;
        }
        if (mask[idx]) {
            int write_idx = vals[i] - 1;
            circle_list[write_idx] = idx % n_circle;
        }
    }
    __syncthreads();

    int tile_offset = blockIdx.x * threads_per_block;
    int t_x = threadId % TILE_WIDTH;
    int t_y = threadId / TILE_WIDTH;

    int num_tiles_x = CEIL_DIV(width, TILE_WIDTH);
    int tile_x = blockIdx.x % num_tiles_x;
    int tile_y = blockIdx.x / num_tiles_x;
    float x = tile_x * TILE_WIDTH + t_x;
    float y = tile_y * TILE_HEIGHT + t_y;
    int pixel_idx = y * width + x;

    if (pixel_idx >= width * height) {
        return;
    }

    float pixel_red = img_red[pixel_idx] == 0 ? 1.0f : img_red[pixel_idx];
    float pixel_green = img_green[pixel_idx] == 0 ? 1.0f : img_green[pixel_idx];
    float pixel_blue = img_blue[pixel_idx] == 0 ? 1.0f : img_blue[pixel_idx];

    for (int32_t i = 0; i < num_block_circles; i++) {
        // if (threadId == 0 && width == 256 && height == 256) {
        //     printf("tile (%d, %d) circle %d/%d\n", tile_x, tile_y, i,
        //     num_block_circles);
        // }
        int c_idx = circle_list[i];
        float c_x = circle_x[c_idx];
        float c_y = circle_y[c_idx];
        float c_radius = circle_radius[c_idx];
        float pixel_alpha = circle_alpha[c_idx];

        // check if pixel is in circle
        float dx = x - c_x;
        float dy = y - c_y;

        if (!(dx * dx + dy * dy < c_radius * c_radius)) {
            continue;
        }

        pixel_red = circle_red[c_idx] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
        pixel_green =
            circle_green[c_idx] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
        pixel_blue = circle_blue[c_idx] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
    }

    img_red[pixel_idx] = pixel_red;
    img_green[pixel_idx] = pixel_green;
    img_blue[pixel_idx] = pixel_blue;
}

__global__ void upstream_reduction(
    int32_t n_circles,
    uint32_t *circle_to_tile_count, // pointer to GPU memory
    uint32_t *blocksums) {

    extern __shared__ __align__(16) uint32_t shmem_raw[]; // OK
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    int threadId = threadIdx.x;
    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * CIRCLES_PER_THREAD_SCAN;

    // load from global memory & perform thread reduction
    uint32_t vals[CIRCLES_PER_THREAD_SCAN];
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        int idx = block_offset + threadId * CIRCLES_PER_THREAD_SCAN + i;
        if (idx >= n_circles) {
            vals[i] = 0;
        } else {
            vals[i] = circle_to_tile_count[idx];
        }
    }

    uint32_t thread_total = 0;
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        thread_total = thread_total + vals[i];
    }

    shmem[threadId] = thread_total;

    // scan across shmem (across all warps in the block)
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        uint32_t cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = shmem[threadId - i] + cur_val;
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    // add prev warp reduction to each thread
    uint32_t threadPrefix = 0;
    if (threadId > 0) { // mask first warp
        threadPrefix = shmem[threadId - 1];
    }
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        vals[i] = threadPrefix + vals[i];
    }

    // write blockSum to workspace array
    if (threadId == threads_per_block - 1) {
        blocksums[blockIdx.x] = shmem[threads_per_block - 1];
    }
}

__global__ void spine_scan(
    uint32_t *blocksums // pointer to GPU memory
) {
    extern __shared__ __align__(16) uint32_t shmem_raw[]; // OK
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    int threads_per_block = blockDim.x;
    int threadId = threadIdx.x;

    uint32_t vals[SPINE_VALS_PER_THREAD];
    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        int idx = threadId * SPINE_VALS_PER_THREAD + i;
        vals[i] = blocksums[idx];
    }

    for (int i = 1; i < SPINE_VALS_PER_THREAD; i++) {
        vals[i] = vals[i - 1] + vals[i];
    }

    uint32_t thread_sum = vals[SPINE_VALS_PER_THREAD - 1];
    shmem[threadId] = thread_sum;

    // scan across shmem
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        uint32_t cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = shmem[threadId - i] + cur_val;
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    uint32_t threadPrefix = 0;
    if (threadId > 0) {
        threadPrefix = shmem[threadId - 1];
    }

    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        vals[i] = threadPrefix + vals[i];
    }

    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        int idx = threadId * SPINE_VALS_PER_THREAD + i;
        blocksums[idx] = vals[i];
    }
}

__global__ void downstream_scan(
    uint32_t *circle_to_tile_count, // pointer to GPU memory
    uint32_t *blocksums             // pointer to GPU memory
) {
    extern __shared__ __align__(16) uint32_t shmem_raw[]; // OK
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * CIRCLES_PER_THREAD_SCAN;

    int threadId = threadIdx.x;

    // load from global memory & perform thread scan
    uint32_t vals[CIRCLES_PER_THREAD_SCAN];
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        int idx = block_offset + threadId * CIRCLES_PER_THREAD_SCAN + i;
        vals[i] = circle_to_tile_count[idx];
    }
    for (int i = 1; i < CIRCLES_PER_THREAD_SCAN; i++) {
        vals[i] = vals[i - 1] + vals[i];
    }

    uint32_t thread_sum = vals[CIRCLES_PER_THREAD_SCAN - 1];
    shmem[threadId] = thread_sum;

    // scan across shmem (across all warps in the block)
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        uint32_t cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = shmem[threadId - i] + cur_val;
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    // fix up each thread in block
    uint32_t threadPrefix = 0;
    if (threadId > 0) { // mask first warp
        threadPrefix = shmem[threadId - 1];
    }
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        vals[i] = threadPrefix + vals[i];
    }

    // add block prefix
    uint32_t block_prefix = 0;
    if (blockIdx.x > 0) {
        block_prefix = blocksums[blockIdx.x - 1];
    }
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        int idx = block_offset + threadId * CIRCLES_PER_THREAD_SCAN + i;
        vals[i] = block_prefix + vals[i];
    }

    // write back to x
    for (int i = 0; i < CIRCLES_PER_THREAD_SCAN; i++) {
        int idx = block_offset + threadId * CIRCLES_PER_THREAD_SCAN + i;
        circle_to_tile_count[idx] = vals[i];
    }
}

__global__ void build_circle_tile_pairs(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    uint32_t *circle_to_tile_count,
    uint32_t *circle_ids,
    uint32_t *tile_ids,
    float const *circle_x, // pointer to GPU memory
    float const *circle_y, // pointer to GPU memory
    float const *circle_radius) {

    int32_t circle_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (circle_idx >= n_circle) {
        return;
    }

    float c_x = circle_x[circle_idx];
    float c_y = circle_y[circle_idx];
    float c_radius = circle_radius[circle_idx];

    int32_t num_tiles_x = CEIL_DIV(width, TILE_WIDTH);

    // get bbox coordinates in tile space
    int box_x_min = max(int((c_x - c_radius) / TILE_WIDTH), 0);
    int box_x_max = min(int((c_x + c_radius) / TILE_WIDTH), (width - 1) / TILE_WIDTH);
    int box_y_min = max(int((c_y - c_radius) / TILE_HEIGHT), 0);
    int box_y_max = min(int((c_y + c_radius) / TILE_HEIGHT), (height - 1) / TILE_HEIGHT);

    int start_idx = circle_idx == 0 ? 0 : circle_to_tile_count[circle_idx - 1];
    int end_idx = circle_to_tile_count[circle_idx];

    int write_idx = start_idx;
    for (int box_y = box_y_min; box_y <= box_y_max; box_y++) {
        for (int box_x = box_x_min; box_x <= box_x_max; box_x++) {
            circle_ids[write_idx] = circle_idx;
            tile_ids[write_idx] = box_y * num_tiles_x + box_x;
            write_idx++;
        }
    }
}

__global__ void count_circle_in_tile(
    int32_t num_pairs,
    int32_t num_tiles,
    uint32_t *circle_ids,
    uint32_t *tile_ids,
    uint32_t *tile_to_circle_count) {

    // each block counts all circles

    extern __shared__ __align__(16) uint32_t shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw); // len(num_tiles)
    int threadId = threadIdx.x;
    int threads_per_block = blockDim.x;
    // initialize shmem
    for (int i = threadId; i < num_tiles; i += threads_per_block) {
        shmem[i] = 0;
    }
    __syncthreads();

    // count circles in tiles
    int start_pair_idx = blockIdx.x * threads_per_block + threadId;
    int end_pair_idx = min(start_pair_idx + threads_per_block, num_pairs);
    for (int idx = start_pair_idx; idx < end_pair_idx; idx += threads_per_block) {
        uint32_t tile_id = tile_ids[idx];
        atomicAdd(&shmem[tile_id], 1);
    }
    __syncthreads();

    // write back to global memory
    for (int i = threadId; i < num_tiles; i += threads_per_block) {
        atomicAdd(&tile_to_circle_count[i], shmem[i]);
    }
}

__global__ void reorder(
    int32_t num_pairs,
    uint32_t const *circle_ids,
    uint32_t const *tile_ids,
    uint32_t *sorted_circle_ids,
    uint32_t *sorted_tile_ids,
    uint32_t const *circle_to_tile_count,
    uint32_t const *tile_to_circle_count) {

    extern __shared__ __align__(16) uint32_t shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw); // len(num_tiles)

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs)
        return;
}

__global__ void render_circles(
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
    uint32_t *circle_ids,
    uint32_t *tile_ids,
    uint32_t *circle_to_tile_count) {
    // int tile_x = blockIdx.x;
    // int tile_y = blockIdx.y;
    // int tile_id = tile_y * num_tiles_x + tile_x;

    // int pixel_x = tile_x * TILE_WIDTH + threadIdx.x;
    // int pixel_y = tile_y * TILE_HEIGHT + threadIdx.y;

    // if (pixel_x >= width || pixel_y >= height)
    //     return;

    // // Initialize tile background to white
    // int32_t pixel_idx = pixel_y * width + pixel_x;
    // img_red[pixel_idx] = 1.0f;
    // img_green[pixel_idx] = 1.0f;
    // img_blue[pixel_idx] = 1.0f;

    // int start_idx = circle_idx == 0 ? 0 : circle_to_tile_count[circle_idx - 1];
    // int end_idx = circle_to_tile_count[circle_idx];
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

    size_t mask_size = num_tiles * CIRCLE_BATCH_SIZE * sizeof(int8_t);
    int8_t *mask = reinterpret_cast<int8_t *>(memory_pool.alloc(mask_size));
    dim3 gridDim(CEIL_DIV(CIRCLE_BATCH_SIZE, CIRCLES_PER_BLOCK));
    dim3 blockDim(CIRCLES_PER_BLOCK);

    // for (int circle_start = 0; circle_start < n_circle;
    //      circle_start += CIRCLE_BATCH_SIZE) {
    int circle_start = 0;
    int circle_end = MIN(circle_start + CIRCLE_BATCH_SIZE, n_circle);
    int batch_size = circle_end - circle_start;

    compute_tile_mask<<<gridDim, blockDim>>>(
        width,
        height,
        batch_size,
        num_tiles_x,
        num_tiles_y,
        circle_x + circle_start,
        circle_y + circle_start,
        circle_radius + circle_start,
        mask);

    dim3 gridDim2(num_tiles);
    dim3 blockDim2(THREADS_PER_BLOCK);
    size_t shmem_size_bytes =
        THREADS_PER_BLOCK * SPINE_VALS_PER_THREAD * sizeof(uint32_t) +
        MIN(n_circle, CIRCLE_BATCH_SIZE) * sizeof(uint32_t);

    if (width == 256) {
        printf(
            "launching with gridDim2=%d, blockDim2=%d, shmem_size_bytes=%zu\n",
            gridDim2.x,
            blockDim2.x,
            shmem_size_bytes);
    }

    tile_scan_and_render<<<gridDim2, blockDim2, shmem_size_bytes>>>(
        width,
        height,
        batch_size,
        mask,
        circle_x + circle_start,
        circle_y + circle_start,
        circle_radius + circle_start,
        circle_red + circle_start,
        circle_green + circle_start,
        circle_blue + circle_start,
        circle_alpha + circle_start,
        img_red,
        img_green,
        img_blue);
    // }
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
