#include <iostream>
#include <functional>
#include <random>
#include <fstream>
#include <cmath>
#include <cstring>
#include <math.h>

#include <cuda_runtime.h>

#include "mh_cpu.hpp"
#include "row.hpp"

using namespace std;

/**
 * Computes the stationary distribution of A (as described in the project
 * proposal).
 * 
 * This is the primary function that will be parallelized, since computing
 * the target distribution can often be expensive.
 **/
double* I_distribution(int N, function<double(double*, int)> target, int m, 
            double* theta_star) {
        double* dist = new double[N + 1];
        double norm = 0;
    
        for (int j = 0; j < N + 1; j++) {
            dist[j] = target(&theta_star[row(j, m)], m);
            norm += dist[j];
        }

        for (int j = 0; j < N + 1; j++) {
            dist[j] /= norm;
        }
        return dist;
    }

/**
 *  Sample from the stationary distribution of A, as computed by
 *  I_distribution()
 **/
int sampleI(double* Idist, int N) {
    uniform_real_distribution<double> unif(0, 1);
    std::random_device rd; 
    static std::mt19937 gen(rd());

    double x = unif(gen);
    double s = 0;
    for (int i = 0; i < N + 1; i++) {
        s += Idist[i];
        if (x < s) {
            return i;
        }
    }

    return 0;
}

double* metropolis_hastings_parallel_cpu(function<double(double*, int)> target,
            function<double*(double*, int)> sample_proposal, int T, 
            double* init, int m, int N) {
    double* samples = new double[T*m];
    cudaEvent_t start_cpu, stop_cpu;
    float cpu_time_milliseconds;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventRecord(start_cpu);

    memcpy(samples, init, m*sizeof(double));
    for (int t = 1; t < T; t = t + N) {
        double* theta_star = new double[(N+1)*m];
        // Sampling can also be parallelized, although the complexity of this
        // step is dwarfed by the step below in practice.
        memcpy(theta_star, &samples[row(t - 1, m)], m*sizeof(double));
        for (int i = 1; i < N + 1; i++) {
            double* proposal = sample_proposal(&theta_star[0], m);
            memcpy(&theta_star[row(i, m)], proposal, m*sizeof(double));
        }

        // The GPU version will parallelize this step here, by computing
        // the distribution in parallel
        double* Idist = I_distribution(N, target, m, 
            theta_star);

        // And also parallelize the sampling of the proposals.
        for (int i = 0; i < N; i++) {
            int I = sampleI(Idist, N);
            if (t + i >= T) {
                break;
            }
            memcpy(&samples[row(t + i, m)], &theta_star[row(I, m)], m*sizeof(double));
        }

        delete [] Idist;
        delete [] theta_star;
    }

    // Stop timer
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&cpu_time_milliseconds, start_cpu, stop_cpu);
    std::cout << "Total CPU Time: " << cpu_time_milliseconds << " ms" << endl;

    return samples;
} 