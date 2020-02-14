//
// Created by James Noeckel on 1/22/20.
//

#pragma once
#include <Eigen/Dense>
#include <vector>

class GaussianMixture {
public:
    /**
     * Construct empty GMM model; must call setComponents() or initialize() before calling learn()
     */
    GaussianMixture();

    /**
     * @param k number of mixture components
     * @param d dimension of data
     */
    GaussianMixture(int k, int d);

    /**
     * Provide initial values for parameters
     * @param means (k, d) matrix, row k is the kth mean
     * @param sigmas vector of k dxd positive definite covariance matrices
     * @param probabilities length k vector of probability weights for each component
     * @return true if initialization was successful
     */
    bool initialize(const Eigen::Ref<const Eigen::MatrixXd> &means, const std::vector<Eigen::MatrixXd> &sigmas, const Eigen::Ref<const Eigen::VectorXd> &probabilities);

    /**
     * Provide initial values for just the means
     * @param means
     * @return true if initialization was successful
     */
    bool initialize(const Eigen::Ref<const Eigen::MatrixXd> &means);

    /**
     * Initialize evenly distributed means in the specified range, with sigmas based on the range.
     * @param lower_bound minimum value of each component of the means
     * @param upper_bound maximum value of each component of the means
     * @param components if not already set, specify number of components
     */
    bool initialize(const Eigen::Ref<const Eigen::VectorXd> &lower_bound, const Eigen::Ref<const Eigen::VectorXd> &upper_bound, int components=-1);

    /**
     * Set parameters to uninitialized, so that the next call to learn() will randomly initialize
     */
    void clear();

    /**
     * Sets the number of components of the model
     * @param k > 0
     */
    void setComponents(int k);

    /**
     * @return number of components k
     */
    int numComponents() const;

    /**
     * @return number of dimensions d
     */
    int numDims() const;

    /**
     * @return (k, d) matrix, each row is that component's mean
     */
    Eigen::MatrixXd means() const;

    /**
     * @return vector of (d, d) covariance matrices
     */
    std::vector<Eigen::MatrixXd> sigmas() const;

    /**
     * @return vector of k log probabilities of each component
     */
    Eigen::VectorXd log_probs() const;

    /**
     * Returns true if the model is complete, i.e. learning succeeded without numerical issues (or model was manually set)
     */
    bool success() const;

    /**
     * Use the currently set parameters for inference (use to bypass learn() as a means of initializing the model)
     * @return true if the resulting model is valid
     */
    bool useCurrentModel();

    /**
     * Learn the parameters of the mixture model from the given data. The first call will use random initialization,
     * and subsequent calls will optimize for the likelihood of data starting from the existing parameters. Note that if
     * appending new data is desired, previous data must still be included in `data`.
     * @param data N x D matrix, where N is the number of observations and D is the dimension
     * @param maxiters maximum number of iterations
     * @param eps minimum change in log likelihood to continue iterating
     * @return number of iterations taken
     */
    int learn(const Eigen::Ref<const Eigen::MatrixXd> &data, int maxiters=200, double eps=1e-3);

    /**
     * Get the log likelihood of `data` for each component. Must have run learn() first.
     * @param data (n, d) matrix where each row is an observation
     * @return log likelihoods as (n, k) matrix where entry i,j is log(P(z_i=j|x_i,mu,sigma,pi))
     */
    Eigen::MatrixXd getLogLikelihoods(const Eigen::Ref<const Eigen::MatrixXd> &data) const;

    /**
     * Get the log likelihood of `data` given the model. Must have run learn() first.
     * @param data (n, d) matrix where each row is an observation
     * @return log likelihood as a length n vector where entry i is log(P(X_i|mu,sigma,pi))
     */
    Eigen::VectorXd getLogLikelihood(const Eigen::Ref<const Eigen::MatrixXd> &data) const;

private:
    /**
     * Set the model components and dimension, and allocate the necessary buffers
     * @param k
     * @param d
     */
    void allocate(int k, int d);
    bool recompute_normalizations();
    /**
     * Execute one step of EM, return the log likelihood given `data` and parameters
     */
    double step(const Eigen::Ref<const Eigen::MatrixXd> &data, bool &success);
    Eigen::MatrixXd getIndividualLogLikelihoods(const Eigen::Ref<const Eigen::MatrixXd> &data) const;

    // fields and parameters
    bool initialized_means_ = false;
    bool initialized_sigmas_ = false;
    bool initialized_pis_ = false;
    bool complete_ = false;
    int d_, k_;
    //parameters and cached derivative quantities
    Eigen::MatrixXd mu_;
    std::vector<Eigen::MatrixXd> sigmas_;
    Eigen::VectorXd log_pi_;
    Eigen::VectorXd normalizations_;
    std::vector<Eigen::LDLT<Eigen::MatrixXd>> cholesky_decompositions_;

    //intermediate buffer during learning
    Eigen::MatrixXd centered_;
    Eigen::MatrixXd log_likelihood_all_;
    Eigen::MatrixXd log_p_z_;

    friend std::ostream &operator<<(std::ostream &o, const GaussianMixture &s);
};

std::ostream &operator<<(std::ostream &o, const GaussianMixture &s);
