#include <functional> 
#include <fstream>
#include <iostream>
#include <random>

#include "mh_gpu.hpp"
#include "mh_cpu.hpp"
#include "row.hpp"

using namespace std;

/* Checks the passed-in arguments for validity. */
void check_args(int argc, char **argv) {
    if (argc != 4) {
        cerr << "Incorrect number of arguments.\n";
        cerr << "Arguments: <threads per block> <max number of blocks> <num samples>\n";
        exit(EXIT_FAILURE);
    }
}

/**
 * An example target distribution. Looks like a pyramid on the [0, 1] x [0, 1]
 * unit square.
 **/
double target_dist(double* theta, int m) {
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
 * An example proposal distribution for the gaussian random walk.
 **/
double* sample_proposal_distribution(double* theta, int m) {
    double* sample = new double[m];

    double std = 1;
    std::random_device rd; 
    static std::mt19937 gen(rd());
    std::normal_distribution<double> norm(0, std);

    for (int i = 0; i < m; i++) {
        sample[i] = theta[i] + norm(gen);
    }

    return sample;
}

int main(int argc, char **argv) {
    check_args(argc, argv);
    int threads_per_block = atoi(argv[1]);
    int blocks = atoi(argv[2]);
    int T = atoi(argv[3]);

    int m = 2;
    double* init = new double[m];
    init[0] = 0.5;
    init[1] = 0.5;

    int burn_in = T / 4;
    int N = blocks * threads_per_block;
    
    // CPU algorithm

    cout << "RUNNING CPU METROPOLIS-HASTINGS..." << endl;

    double* samples = metropolis_hastings_parallel_cpu(target_dist, 
        sample_proposal_distribution, T, init, 2, 1);

    ofstream fout_cpu;
    fout_cpu.open("samples_cpu.txt");
    for (int i = burn_in; i < T; i++) {
        fout_cpu << (&samples[row(i, m)])[0] << " " << (&samples[row(i, m)])[1] << endl;
    }
    fout_cpu.close();

    delete [] samples;

    // GPU algorithm

    cout << "RUNNING GPU METROPOLIS-HASTINGS..." << endl;

    samples = metropolis_hastings_gpu(sample_proposal_distribution,
        T, init, m, N, blocks, threads_per_block);

    ofstream fout_gpu;
    fout_gpu.open("samples_gpu.txt");
    for (int i = burn_in; i < T; i++) {
        fout_gpu << (&samples[row(i, m)])[0] << " " << (&samples[row(i, m)])[1] << endl;
    }
    fout_gpu.close();

    delete [] samples;
    return 0;
}
