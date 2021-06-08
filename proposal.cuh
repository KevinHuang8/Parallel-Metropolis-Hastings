#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

/**
* Run one iteration of the parallel MH algorithm on GPU.
* sample_host - the (T + 1) x m array of samples on the host in which to
* store the output
* prev_sample - a length m array which holds the last sample from the previous
* iteration
* m - dimensionality of the parameter space
*/
void call_kernels(const unsigned int blocks,
    const unsigned int threads_per_block,
    double* sample_host,
    double* prev_sample,
    int m);