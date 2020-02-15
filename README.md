## Overview

This is an implementation of the expectation maximization (EM) algorithm for maximum-likelihood estimation of Gaussian mixture model parameter estimation. It supports data of arbitrary dimensions.
It is based on the Eigen linear algebra library.

## Example Usage
The below snippet initializes a Gaussian mixture model with 3 components in 2 dimensions, then optimizes the parameters for 50 random observations and reports the likelihood of a new observation given the learned model.
```c++
GaussianMixture gmm(3, 2);
Eigen::MatrixXd data = Eigen::MatrixXd::Random(50, 2);
int iters = gmm.learn(data);
std::cout << "took " << iters << " iterations" << std::endl;
std::cout << "-------- parameters --------" << std::endl << gmm << std::endl;

Eigen::RowVector2d point(0, 0);
std::cout << "log likelihood at point " << point << ": " << gmm.getLogLikelihood(point) << std::endl;
```

The below example shows how to manually set the parameters of the model and use it as a likelihood function:
```c++
std::vector<Eigen::MatrixXd> cov_default(1, Eigen::MatrixXd::Identity(3, 3));
Eigen::Matrix<double, 1, 1> pi(1);
Eigen::Matrix<double, 1, 2> mean(1, -1);
gmms[i].initialize(mean, cov_default, pi);
gmms[i].useCurrentModel();
Eigen::RowVector2d point(0, 0);
std::cout << "log likelihood at point " << point << ": " << gmm.getLogLikelihood(point) << std::endl;
```
