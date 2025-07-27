#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>
#include "MLP.hpp"
#include "layers.hpp"
#include "mnist_loader.hpp"
#include "trainer.hpp"

typedef float data_t;

Trainer::Trainer(MLP& model, CrossEntropyLoss& loss_fn) : model(model), loss_fn(loss_fn) {}

void Trainer::train(std::vector<MNISTSample> &dataset, int epochs, float lr) {
    std::cout << "Starting Training Loop over " << epochs << " epochs" << std::endl;

    /* Main training Loop over all epoths */
    for (int i = 0; i < epochs; i++) {
        std::cout << "Epoch 1: " << std::endl;
        data_t loss = 0;
        std::vector<data_t> logits;
        std::vector<data_t> grad_logits;

        /* Gradient Decent over whole dataset */
        for(int j = 0; j < dataset.size(); j++){
            if (j % 300 == 0) {std::cout << "Sample " << j << " : " << std::endl;}
            logits = model.forward(dataset[j].pixels);
            if (j % 300 == 0) {std::cout << "Sample " << j << " Forward Finished! " << std::endl;}
            data_t sample_loss = loss_fn.forward(logits, dataset[j].label);
            loss += sample_loss;
            if (j % 300 == 0) {std::cout << "Sample Loss: " << sample_loss << std::endl; }
            grad_logits = loss_fn.backward(dataset[j].label);
            model.backward(grad_logits);
            model.update(lr);
        }

        std::cout << "EPOCH " << i << " Average Loss : " << loss/dataset.size() << std::endl << std::endl;
        std::cout << "==================================" << std::endl;
    }
}