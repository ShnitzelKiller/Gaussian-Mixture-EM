//
// Created by James Noeckel on 1/27/20.
//

#pragma once
#include <Eigen/Dense>
#include <random>

struct NormalRandomVariable
{
    explicit NormalRandomVariable(Eigen::Ref<const Eigen::MatrixXd> const& covar);

    NormalRandomVariable(Eigen::Ref<const Eigen::VectorXd> const& mean, Eigen::Ref<const Eigen::MatrixXd> const& covar);

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const;
};