#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <memory>
#include "MLP.hpp"
#include "layers.hpp"

typedef float data_t;

void MLP::add(Layer *layer) {
    /* Converts the raw pointer to unique_ptr and pushes to back of layers vector */
    layers.push_back(std::unique_ptr<Layer>(layer));
}

std::vector<data_t> MLP::forward(const std::vector<data_t> &input) {
    std::vector<data_t> out = input;    /* Maintain scope of out after loop */

    for (auto& layer : this->layers) {
        out = layer->forward(out);
    }
    return out;         /* logits for softmax */
}

std::vector<data_t> MLP::backward(const std::vector<data_t>& grad_output) {
    std::vector<data_t> grad = grad_output;
    
    /* Loop backwards from last layer to first */
    for (std::vector<std::unique_ptr<Layer>>::reverse_iterator it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

void MLP::update(float lr) {
    for (auto& layer: layers) {
        layer->update(lr);
    }
}