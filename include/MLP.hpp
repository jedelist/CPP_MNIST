#pragma once
#include <iostream>
#include <vector>
#include "layers.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <memory>

typedef float data_t;

class MLP {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        float lr;

    public:
        MLP(float lr);
        void add(Layer *layer);
        std::vector<data_t> forward(const std::vector<data_t>& input);
        std::vector<data_t> backward(const std::vector<data_t>& grad_output);
        void update(float lr);
        void save(const std::string& filename) const;
        void load(const std::string& filename);
};