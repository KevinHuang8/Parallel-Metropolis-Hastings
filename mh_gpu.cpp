#include <iostream>
#include <functional>
#include <random>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <math.h>
#include <cstring>

#include <cuda_runtime.h>
#include "proposal.cuh"
#include "mh_gpu.hpp"
#include "row.hpp"

using namespace std;

double* metropolis_hastings_gpu(
    function<double*(double*, int)> sample_proposal,
    int T, double* init, int m, int N, int blocks, int threads_per_block) {
        double* samples = new double[(T+N)*m];
        cudaEvent_t start_gpu, stop_gpu;
        float gpu_time_milliseconds;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        cudaEventRecord(start_gpu);

        memcpy(samples, init, m * sizeof(double));
        for (int t = 1; t < T; t = t + N) {

            call_kernels(blocks, threads_per_block,
                &samples[row(t, m)], &samples[row(t - 1, m)], m);

        }
       
    // Stop timer
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&gpu_time_milliseconds, start_gpu, stop_gpu);

    std::cout << "Total GPU Time: " << gpu_time_milliseconds << " ms" << endl;
    return samples;
} 