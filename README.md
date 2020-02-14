This is an implementation of the expectation maximization (EM) algorithm for maximum-likelihood estimation of Gaussian mixture model parameter estimation. It supports data of arbitrary dimensions.
It is based on the Eigen linear algebra library.

## Example Usage

```c++
GaussianMixture gmm(3, 2);
Eigen::MatrixXd data = Eigen::MatrixXd::Random(50, 2);
int iters = gmm.learn(data);
std::cout << "took " << iters << " iterations" << std::endl;
std::cout << "-------- parameters --------" << std::endl << gmm << std::endl;

Eigen::RowVector2d point(0, 0);
std::cout << "log likelihood at point " << point << ": " << gmm.getLogLikelihood(point) << std::endl;
```
