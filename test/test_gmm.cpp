//
// Created by James Noeckel on 1/22/20.
//

#include "GaussianMixture.h"
#ifdef USE_OPENCV
#include <opencv2/opencv.hpp>
#endif
#include <ctime>
#include "NormalRandomVariable.h"
#include <iostream>

#define trials 100
#define width 500
#define height 500

#ifdef USE_OPENCV
void display_gmm(const Eigen::Ref<const Eigen::MatrixX2d> &data, const GaussianMixture &gmm, const std::string &filename) {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC3);
    Eigen::MatrixX2i coords(width * height, 2);
    for (int i = 0; i < width * height; i++) {
        coords(i, 0) = i / width;
        coords(i, 1) = i % width;
    }
    Eigen::VectorXd total_log_likelihoods = gmm.logp_data(coords.cast<double>());
    double max_log_likelihood = total_log_likelihoods.maxCoeff();

    for (int i = 0; i < width * height; i++) {
        auto color = static_cast<unsigned char>(255 * exp(total_log_likelihoods(i) - max_log_likelihood));
        mask.at<cv::Vec3b>(coords(i, 0), coords(i, 1)) = cv::Vec3b(color, color, color);
    }
    Eigen::MatrixXd log_likelihoods = gmm.logp_z_given_data(data);
    for (int i = 0; i < data.rows(); i++) {
        int j;
        log_likelihoods.row(i).maxCoeff(&j);
        cv::Point2i point(static_cast<int>(data(i, 1)), static_cast<int>(data(i, 0)));
        if (point.x >= 0 && point.x < width && point.y >= 0 && point.y < height) {
            if (j == 0) {
                mask.at<cv::Vec3b>(point) = cv::Vec3b(255, 0, 255);
            } else if (j == 1) {
                mask.at<cv::Vec3b>(point) = cv::Vec3b(0, 255, 255);
            } else if (j == 2) {
                mask.at<cv::Vec3b>(point) = cv::Vec3b(0, 0, 255);
            } else {
                mask.at<cv::Vec3b>(point) = cv::Vec3b(255, 255, 0);
            }
        }
    }
    cv::imwrite(filename, mask);
}
#endif

GaussianMixture test_gmm(bool use_kmeans, Eigen::MatrixX2d &out_data, bool print_comparison) {
    GaussianMixture gmm;
    int k = 4;
    gmm.setComponents(k);
    int n = 2000;
    std::cout << "generating data..." << std::endl;

    Eigen::MatrixX2d data(n, 2);
    Eigen::MatrixX2d means(k, 2);
    means << 250, 250,
             100, 250,
             250, 100,
             400, 400;
    std::vector<Eigen::Matrix2d> sigmas{
            (Eigen::Matrix2d() << 1000, 500, 500, 750).finished(),
            Eigen::Matrix2d::Identity() * (50 * 50),
            (Eigen::Matrix2d() << 500, 0, 0, 250).finished(),
            (Eigen::Matrix2d() << 500, 0, 0, 500).finished()
    };
    std::vector<NormalRandomVariable> randvs;
    randvs.reserve(k);
    for (int i=0; i<k; i++) {
        randvs.emplace_back(means.row(i).transpose(), sigmas[i]);
    }
    for (int i = 0; i < n; i++) {
        int cluster = std::rand() % k;
        data.row(i) = randvs[cluster]().transpose();
    }
    std::cout << "generated data " << std::endl;

    std::cout << "GMM parameter estimation..." << std::endl;
    float time_avg = 0.0f;
    int total_iters = 0;
    int total_successes = 0;
    bool warning_printed = false;
    for (int i = 0; i < trials; i++) {
        gmm.clear();
        if (!use_kmeans) {
            gmm.initialize_random_means(data);
        }
        auto start_t = clock();
        int iters = gmm.learn(data, 200);
        total_iters += iters;
        auto total_t = clock() - start_t;
        float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
        time_avg += time_sec;
        if (gmm.success()) {
            total_successes += 1;
        }
    }
    time_avg /= trials;
    std::cout << "average time: " << time_avg << std::endl;
    std::cout << "average iters: " << total_iters / static_cast<float>(trials) << std::endl;
    std::cout << "total successes: " << total_successes << '/' << trials << std::endl;
    if (print_comparison) {
        std::cout << "---- parameter comparison ----" << std::endl
                  << "means:            ground truth (not in same order):" << std::endl;
        for (int i = 0; i < k; i++) {
            std::cout << '[' << gmm.means().row(i) << "] -------- [" << means.row(i) << ']' << std::endl;
        }
        std::cout << "covariances:      ground truth (not in same order):" << std::endl;
        std::cout << "------------------------------------" << std::endl;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < 2; j++) {
                std::cout << '|' << gmm.sigmas()[i].row(j) << "| -------- |" << sigmas[i].row(j) << '|' << std::endl;
            }
            std::cout << "------------------------------------" << std::endl;
        }
    }
    out_data = std::move(data);
    return gmm;
}

void show_eigenvalues(const GaussianMixture &gmm) {
    for (int i=0; i<gmm.numComponents(); i++) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(gmm.sigmas()[i]);
        std::cout << "eigenvalues for cluster " << i << ": [" << eig.eigenvalues().transpose() << "]; mean: [" << gmm.means().row(i) << ']' << std::endl;
        std::cout << "eigenvectors:\n" << eig.eigenvectors() << std::endl;
    }
}

int main(int argc, char **argv) {
    auto seed = (unsigned int) time(nullptr);
    std::cout << "seed: " << seed << std::endl;
    srand(seed);
    {
        std::cout << "=======testing without kmeans=======" << std::endl;
        Eigen::MatrixX2d data;
        test_gmm(false, data, false);
        std::cout << "=======testing with kmeans=======" << std::endl;
        GaussianMixture gmm = test_gmm(true, data, true);
#ifdef USE_OPENCV
        display_gmm(data, gmm, "test_gmm_1.png");
#endif
    }

    {
        std::cout << "=========== testing 1D subspaces setting min variance to 2 ===========" << std::endl;
        GaussianMixture gmm(3, 2, 2);
        Eigen::MatrixX2d data(20, 2);
        for (int i=0; i<10; i++) {
            data.row(i) = Eigen::RowVector2d(200+i*10, 200+i*15);
            data.row(i+10) = Eigen::RowVector2d(500-i*30, 200 + i*10);
        }
        int iters = gmm.learn(data);
        if (!gmm.success()) {
            std::cout << "learning on disjoint linear subspace data failed" << std::endl;
            return 1;
        } else {
            std::cout << "succeeded in " << iters << " iterations" << std::endl;
        }
        show_eigenvalues(gmm);
#ifdef USE_OPENCV
        display_gmm(data, gmm, "test_gmm_2.png");
#endif
    }

    {
        std::cout << "=========== testing high degeneracy ===========" << std::endl;
        GaussianMixture gmm(3, 2, 0.001);
        Eigen::MatrixXd data = Eigen::MatrixXd::Zero(50, 2);
        data.row(4) = Eigen::RowVector2d(1, 1);
        Eigen::MatrixXd data_extrapoint(data);
        data_extrapoint.row(19) = Eigen::RowVector2d(.01, -.01);
        data *= 0.01;
        int initialized = gmm.initialize_k_means(data);
        if (initialized != 0) {
            std::cout << "init should have failed" << std::endl;
            return 1;
        }

        initialized = gmm.initialize_k_means(data_extrapoint);
        if (!initialized) {
            std::cout << "initialization failed" << std::endl;
            return 1;
        }
        int iters = gmm.learn(data_extrapoint);
        if (!gmm.success()) {
            std::cout << "learning on degenerate data failed" << std::endl;
            return 1;
        }
        std::cout << "finished; k-means initialization iters: " << initialized << "; learning iters: " << iters << std::endl;
        show_eigenvalues(gmm);
    }
    return 0;
}
