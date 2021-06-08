#include <functional>

/**
 * The parallelizable generalized metropolis hastings algorithm.
 * 
 * Parameters:
 *  target - the pdf of the target distribution.
 *  sample_proposal - a function that takes in a point in parameter space
 *                    and samples a new point from a proposal distribution.
 *  T - number of samples to generate
 *  init - the initial point
 *  m - the number of parameters
 *  N - the number of samples to simultaneously propose at each iteration.
 * 
 * Returns a 2D (flattened) array of samples.
 **/
double* metropolis_hastings_parallel_cpu(std::function<double(double*, int)> target,
            std::function<double*(double*, int)> sample_proposal, int T, 
            double* init, int m, int N);