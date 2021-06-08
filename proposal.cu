#include "proposal.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <math_constants.h>
#include <cstdio>
#include <stdint.h>

#define gpu_errchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

__device__ double atomicAdd_double(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

/**
* The target distribution we are trying to sample from.
*/
__device__
double target_dist(const double* theta, int m) {
    if (theta[1] < theta[0] && theta[1] >= 0 && theta[1] <= 1 - theta[0]) {
      return(6*theta[1]);
    }
    else if (1 - theta[0] <= theta[1] && theta[1] <= theta[0] && theta[0] <= 1) {
      return(3 - 6*(theta[0] - 0.5));
    }
    else if (theta[1] >= theta[0] && theta[1] <= 1 && theta[1] >= 1 - theta[0]) {
      return(3 - 6*(theta[1] - 0.5));
    }
    else if (theta[0] <= theta[1] && theta[1] <= 1 - theta[0] && theta[0] >= 0) {
      return(6*theta[0]);
    }
    else {
      return(0);
    }
  }

/**
 *  Sample from the categorical distribution given by Idist.
 **/
 __device__
 int sampleI(double* Idist, int N, curandState* state) {
     double x = curand_uniform_double(state);
     double s = 0;
     for (int i = 0; i < N + 1; i++) {
         s += Idist[i];
         if (x < s) {
             return i;
         }
     }
     return 0;
 }

__device__
int row_(int j, int m) {
    return m*j;
}

/**
* Initializes random states with seed equal to thread id
*/
__global__
void initialize_curand_kernel(curandState* rand_states) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int seed = clock64();
    curand_init(seed, id, 0, &rand_states[id]);
}

/**
* Computes the stationary distribution of I (as described in README)
*/
__global__
void I_distribution_kernel(const double* theta_star, int N, int m, 
        double* dist) {
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (thread_index < N + 1) {
        // we are using a symmetrical proposal, so no need for calculating the
        // proposal distribution
        dist[thread_index] = target_dist(&theta_star[row_(thread_index, m)], m);
        thread_index += blockDim.x * gridDim.x;
    }
}

/**
* Computes the sum of array dist of size (N + 1), and stores sum in norm
*/
__global__
void sum_kernel(double* dist, double* norm, int N) {
    extern __shared__ double sdata[];

    unsigned int t = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    // first, reduce elements with a stride of blockDim.x*gridDim.x
    // from global memory into shared memory
    sdata[t] = 0;
    while (i < N + 1) {
        sdata[t] = sdata[t] + dist[i];
        i += blockDim.x * gridDim.x;
    }

    // Sequential addressing (#3 from Mark Harris pdf)
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride>>1) {
        __syncthreads();
        if (t < stride) {
            sdata[t] = sdata[t] + sdata[t + stride];
        }
    }

    // sdata[0] holds the sum.
    if (threadIdx.x == 0) {
        atomicAdd_double(norm, sdata[0]);
    }
}

/**
*  Divides each element in dist (size N+1) by norm.
*/
__global__
void
divide_kernel(double* dist, double* norm, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    while (i < N + 1) {
        dist[i] = dist[i] / *norm;

        i += blockDim.x * gridDim.x;
    }
}

/**
*  Use the calculated I distribution dist to get N samples from the N + 1
*  proposals stored in theta_star.
*/
__global__
void sample_kernel(double* dist, double* theta_star, int N, int m, 
        double* sample_gpu, curandState* rand_states) {
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

    int I = sampleI(dist, N, &rand_states[thread_index]);
    memcpy(&sample_gpu[row_(thread_index, m)],
        &theta_star[row_(I, m)], m*sizeof(double));
}

/**
* Generates proposals in parallel.
*/
__global__
void proposal_kernel(double* theta_star, int m) {
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    curandState state;
    int seed = clock64();
    curand_init(seed, thread_index - 1, 0, &state);

    for (int i = 0; i < m; i++) {
        theta_star[row_(thread_index, m) + i] = theta_star[i] 
            + curand_normal_double(&state);
    }
}

/**
* Call the kernels for generating N samples.
* theta_star_host - (N + 1) x m proposal matrix.
* m - dimensionality of the parameter space
*/
void call_kernels(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            double* sample_host,
                            double* prev_sample,
                            int m) {

    double* theta_star_gpu;
    int N = blocks * threads_per_block;

    // Allocate memory and copy over data
    gpu_errchk(cudaMalloc((void **) &theta_star_gpu, (N+1) * m * sizeof(double)));
    // gpu_errchk(cudaMemcpy(theta_star_gpu, theta_star_host,  (N+1) * m * sizeof(double),
    //     cudaMemcpyHostToDevice));
    gpu_errchk(cudaMemcpy(theta_star_gpu, prev_sample,  m * sizeof(double),
        cudaMemcpyHostToDevice));

    // Generate N proposals
    proposal_kernel<<<blocks, threads_per_block>>>(theta_star_gpu, m);

    // Allocate memory for I distribution
    double* dist_out;
    gpu_errchk(cudaMalloc((void **) &dist_out, (N+1) * sizeof(double)));
    gpu_errchk(cudaMemset(dist_out, 0.0, (N+1)*sizeof(double)));

    // Calculate distribution
    I_distribution_kernel<<<blocks, threads_per_block>>>(theta_star_gpu, 
        N, m, dist_out);

    // Normalize the distribution
    double* norm;
    gpu_errchk(cudaMalloc((void**) &norm, sizeof(double)));
    gpu_errchk(cudaMemset(norm, 0, sizeof(double)));

    sum_kernel<<<blocks, threads_per_block, threads_per_block*sizeof(double)>>>(
        dist_out, norm, N
    );

    divide_kernel<<<blocks, threads_per_block>>>(dist_out, norm, N);

    // Allocate memory for curand states
    curandState* rand_states;
    gpu_errchk(cudaMalloc((void**) &rand_states, N * sizeof(curandState)));

    // Initialize random states
    initialize_curand_kernel<<<blocks, threads_per_block>>>(rand_states);

    // Allocate space on device for N samples
    double* sample_gpu;
    gpu_errchk(cudaMalloc((void **) &sample_gpu, N * m * sizeof(double)));

    // Choose N samples
    sample_kernel<<<blocks, threads_per_block>>>(dist_out, theta_star_gpu,
        N, m, sample_gpu, rand_states);

    // Copy output samples to CPU
    gpu_errchk(cudaMemcpy(sample_host, sample_gpu, N*m*sizeof(double), 
        cudaMemcpyDeviceToHost));

    // Free GPU memory
    gpu_errchk(cudaFree(theta_star_gpu));
    gpu_errchk(cudaFree(dist_out));
    gpu_errchk(cudaFree(norm));
    gpu_errchk(cudaFree(rand_states));
    gpu_errchk(cudaFree(sample_gpu));
}
