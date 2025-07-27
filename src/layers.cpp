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

std::vector<data_t> Linear::forward(const std::vector<data_t> &input) {

    /* Convert vector to Eigen::Vector*/
    Eigen::VectorXf x = Eigen::Map<const Eigen::VectorXf>(input.data(), inSize);
    this->last_input = x;

    Eigen::VectorXf y = (W * x) + b;        /* Linear Matrix Multiplication with bias */

    /* Convertin y back go std::vector for output */
    std::vector<data_t> output(y.data(), y.data() + y.size());
    return output;
    
}

std::vector<data_t> Linear::backward(const std::vector<data_t> &grad_output) {
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
ReLU::ReLU(int features) : Layer(features, features), mask(features) {}

std::vector<data_t> ReLU::forward(const std::vector<data_t> &input) {
    Eigen::VectorXf x = Eigen::Map<const Eigen::VectorXf>(input.data(), inSize);
    
    /* Applies max(0.0f, x) to each element */
    Eigen::VectorXf y = x.cwiseMax(0.0f);

    /* Store the mask for dReLU */
    mask = (x.array() > 0.0f).cast<data_t>();

    std::vector<data_t> output(y.data(), y.data() + y.size());
    return output;
}

std::vector<data_t> ReLU::backward(const std::vector<data_t> &grad_output) {
    Eigen::VectorXf grad_out = Eigen::Map<const Eigen::VectorXf>(grad_output.data(), outSize);

    /* dReLU = grad_out * mask */
    Eigen::VectorXf grad_input = grad_out.array() * mask.array();

    std::vector<data_t> grad_in(grad_input.data(), grad_input.data() + grad_input.size());
    return grad_in;
}

/* Softmax forward definition */
std::vector<data_t> CrossEntropyLoss::softmax(const std::vector<data_t> &logits) {
    // Convert to Eigen::VectorXf
    Eigen::VectorXf z = Eigen::Map<const Eigen::VectorXf>(logits.data(), logits.size());

    /* Softmax */
    Eigen::VectorXf exp_z = (z.array() - z.maxCoeff()).exp();
    Eigen::VectorXf out = exp_z.array() / exp_z.sum();

    std::vector<data_t> result(out.data(), out.data() + out.size());
    return result;
}

/* Cross Entropy Loss Forward */
data_t CrossEntropyLoss::forward(const std::vector<data_t> &logits, uint8_t label) {
    Eigen::VectorXf z = Eigen::Map<const Eigen::VectorXf>(logits.data(), logits.size());
    
    /* Softmax with stability */
    Eigen::ArrayXf exp_z = (z.array() - z.maxCoeff()).exp(); 
    Eigen::ArrayXf probs = exp_z / exp_z.sum();

    /* Save for backward */
    last_softmax = std::vector<data_t>(probs.data(), probs.data() + probs.size());
    
    /* Cross entropy = -log(prob of true class) */
    float p_true = probs(label);
    return -std::log(p_true + 1e-12f);  /* gotta avoid log(0) so Epsilon = 1e-12 */
}

std::vector<data_t> CrossEntropyLoss::backward(const std::vector<data_t> &logits, uint8_t label) {
    Eigen::VectorXf probs = Eigen::Map<const Eigen::VectorXf>(last_softmax.data(), last_softmax.size());

    /* dL/dlogits = softmax - 1 at index[label] */
    probs(label) -= 1.0f;       // Eigen has element-wise
    return std::vector<data_t>(probs.data(), probs.data() + probs.size()); // back to std::vector no loss scaling yet
}