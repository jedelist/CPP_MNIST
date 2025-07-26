#include "layers.hpp"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include <random>

typedef float data_t;

/* Xavier Init for the weights (maybe I'll add Kaiming later) */
static float xavier_init(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-limit, limit);
    return dist(gen);
}

/* Constructor */
Linear::Linear(int in_features, int out_features) : 
    Layer(in_features, out_features), 
    W(out_features, in_features),
    b(out_features), 
    grad_W(out_features, in_features),
    grad_b(out_features), 
    last_input(in_features) 
{
    // Xavier initialization weights
    for (int i = 0; i < out_features; i++) {
        for (int j = 0; j < in_features; j++) {
            W(i, j) = xavier_init(in_features, out_features);
        }
        b(i) = 0.0f;
    }
}

std::vector<data_t> Linear::forward(const std::vector<data_t> &input) override {

    /* Convert vector to Eigen::Vector*/
    Eigen::VectorXf x = Eigen::Map<const Eigen::VectorXf>(input.data(), inSize);
    this->last_input = x;

    Eigen::VectorXf y = (W * x) + b;        /* Linear Matrix Multiplication with bias */

    /* Convertin y back go std::vector for output */
    std::vector<data_t> output(y.data(), y.data() + y.size());
    return output;
    
}

std::vector<data_t> Linear::backward(const std::vector<data_t> &grad_output) override {
    Eigen::VectorXf grad_out = Eigen::Map<const Eigen::VectorXf>(grad_output.data(), outSize);

    /* dI/dW = */
    grad_W = grad_out * last_input.transpose(); // outer product
    grad_b = grad_out;

    Eigen::VectorXf grad_input = W.transpose() * grad_out;

    /* Cast back to vector */
    std::vector<data_t> grad_in(grad_input.data(), grad_input.data() + grad_input.size());
    return grad_in;
}

void Linear::update(float lr) {
    W = W - lr * grad_W;
    b = b - lr * grad_b;
}

/* ReLU Constructor calling base constructor */
ReLU::ReLU(int in_features, int out_features) : Layer(in_features, out_features)