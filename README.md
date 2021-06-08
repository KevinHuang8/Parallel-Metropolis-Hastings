# Final Project Writeup/Description

This program implements the parallel Metropolis Hastings algorithm as described here: https://www.pnas.org/content/111/49/17408.

### MH Algorithm Description

For convenience, here is the pseudocode for the parallel MH algorithm that the program implements:

1. Initialize starting point $\tilde x_1$, auxiliary variable $I = 1$, and counter $t = 0$. Let $X_i$ denote the MCMC samples we want to get.
2. Until $t \geq N$:
   1. Sample points $\tilde x_i$ for $i \in [1, N+1]$ excluding $i = I$, drawn from the proposal kernel $K(\tilde x_I, \dot)$. That is, we start at point $\tilde x_I$, and then draw $N$ additional samples from the starting point. By default, this program uses a gaussian random walk, i.e. picks a random normally distributed number with mean $\tilde x_I$ and variance $1$.
   2. Compute the stationary distribution of $I$ conditioned on $\tilde x_{1:N+1}$, which is given by $p(I = j | \tilde x_{I:{N+1}}) \propto \pi(\tilde x_j)$, where $\pi$ is the target distribution. Normally, we have the incorporate the proposal kernel $K$, but since we are using a symmetrical proposal distribution, we don't have to.
   3. Normalize the stationary distribution.
   4. For $i = 1,\dots N$, 
      1. Sample $I \sim p$ (the stationary distribution calculated previously)
      2. Let $X_{t+i} = \tilde x_I$
   5. Update $t = t + N$

The GPU algorithm parallelizes steps 1-4 inside the loop in step 2 by computing each of $N$ proposals on a separate GPU thread.

The CPU sequential algorithm is the special case of the above algorithm where $N = 1$ (no parallelization).

### Usage instructions/Project Description

Use the Makefile to compile.

Run `./main <blocks> <threads per block> <num samples>` to run the program.

The program will run a CPU and a GPU version of the algorithm (the CPU version being slightly changed from the submission to be more efficient), and print out the time taken to run both algorithms.

The cpu code is located in `mh_cpu.cpp`, and the gpu code is located in `mh_gpu.cpp` and `proposal.cu`. The driver is located in `main.cpp`. All functions are documented, with the main functions being documented in the header files.

The algorithm will sample from a 2D target distribution defined by this image:

<img src="images\target_example.png" alt="target_example" style="zoom:67%;" />

This target distribution is an example toy distribution and can be replaced with any distribution defined by the user. Because this target distribution is called on the gpu, it is implementation specific, and so the user must redefine the `target_dist` function in `proposal.cu`. To change the distribution for the cpu version, one can simply pass in a function with the correct signature into the `target` parameter of `metropolis_hastings_parallel_cpu`.

The algorithm starts at point `init`, which is $(0.5, 0.5)$ by default. This is defined in the driver `main.cpp`.

The program will output a list of $T$ points ($T$ given by the third argument to the program) to a file, one on each line, which correspond to the MCMC samples from the target distribution. There is a default burn in period of $T / 4$ which is excluded from the sample list.

The gpu algorithm outputs its points to `samples_gpu.txt`, and the cpu algorithm outputs to `samples_cpu.txt`. 

To visualize the points, a Python notebook called `visualize.ipynb` is included which will graph the points when run. This assumes that the target distribution has 2 parameters and thus can be visualized in 2 dimensions. To sample from distributions with different dimensionality, the user can change `m` in the driver `main.cpp` and update the initial point `init` accordingly to have `m` dimensions.

When the default example distribution samples are visualized by the visualization script, they should look something like this:

<img src="images\output_plot.png" alt="target_example" style="zoom:125%;" />



### Results/Performance Analysis

Speed comparison for different values of $N$, for generating $T = 250000$ samples

| $N$   | CPU time (ms) | GPU time (ms) |
| ----- | ------------- | ------------- |
| 256   | 301.4         | 1177.26       |
| 1024  | 301.4         | 375.60        |
| 4096  | 301.4         | 175.02        |
| 16384 | 301.4         | 138.45        |

As we can see, the GPU can achieve over a 2x speedup compared to the CPU version for this particular target distribution.

*However*, note that the main purpose of parallelization is to parallelize the computation of the target density $\pi$, as in each iteration, we only have to compute $\pi$ once in parallel, but $N$ times sequentially. In many applications, the target density $\pi$ is expensive to compute. However, the example distribution given here is rather trivial to compute. Thus, we would expect a far greater speed up for use cases where $\pi$ is harder to compute. Still, this proof of concept demonstrates that even for trivial $\pi$, the GPU version can have a significant speed up. 

Furthermore, one additional optimization was not implemented in the GPU algorithm. As implemented, the GPU algorithm makes stride $m$ accesses to data as the proposals are stored in row major format, where $m$ is the dimensionality of the parameter space. In our use case, $m = 2$, so this effect is not not that severe, but for high $m$, this can have a significant performance impact. The data can be stored in column major format instead to better take advantage of locality. 

