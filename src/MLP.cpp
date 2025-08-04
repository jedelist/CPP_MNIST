#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <memory>
#include <algorithm>
#include <fstream>
#include "MLP.hpp"
#include "layers.hpp"

typedef float data_t;

MLP::MLP(float lr) : lr(lr) {};

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

/* Calls all update methods in each layer */
void MLP::update(float lr) {
    for (auto& layer: layers) {
        layer->update(lr);
    }
}

void MLP::save(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open model file for saving!");

    out << layers.size() << "\n"; // number of layers

    for (auto& layer : layers) {
        if (auto* lin = dynamic_cast<Linear*>(layer.get())) {
            out << "Linear\n";
            lin->save(out);
        } else if (dynamic_cast<ReLU*>(layer.get())) {
            out << "ReLU\n";
            // no params to save
        }
    }
    out.close();
}

void MLP::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open model file for loading!");

    size_t num_layers;
    in >> num_layers;
    if (num_layers != layers.size()) {
        throw std::runtime_error("Mismatch in number of layers when loading model!");
    }

    for (auto& layer : layers) {
        std::string layer_type;
        in >> layer_type;
        if (layer_type == "Linear") {
            auto* lin = dynamic_cast<Linear*>(layer.get());
            lin->load(in);
        } else if (layer_type == "ReLU") {
            // nothing to load
        }
    }
    in.close();
}

/* Function to Predict or run inference */
int MLP::predict(const std::vector<data_t> & input) {
    std::vector<data_t> logits = forward(input);
    std::vector<data_t> output = CrossEntropyLoss::softmax(logits);

    /* Find index of the max element */
    auto max_it = std::max_element(output.begin(), output.end());
    int index_of_max = std::distance(output.begin(), max_it);

    return index_of_max;
}