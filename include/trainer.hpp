#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include "MLP.hpp"
#include "layers.hpp"
#include "mnist_loader.hpp"

typedef float data_t;

class Trainer {
    private:
        MLP& model;
        CrossEntropyLoss& loss_fn;

    public:
        Trainer(MLP& model, CrossEntropyLoss& loss_fn);
        void train(std::vector<MNISTSample> &dataset, int epochs, float lr);
};