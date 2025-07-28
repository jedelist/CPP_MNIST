#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <string>
#include <fstream>
#include "MLP.hpp"
#include "layers.hpp"
#include "mnist_loader.hpp"
#include "trainer.hpp"

#define EPOCHS 10

typedef float data_t;


int main (int argc, char **argv) {

    /* Load training dataset */
    std::vector<MNISTSample> train_dataset = MNISTLoader::load_training(MNISTLoader::get_data_dir());

    /* Initialize MLP model */
    float lr = 0.0001f;
    MLP model = MLP(lr);

    /* Initialize Layers of the MLP model */
    Layer *input = new Linear(784, 512);
    Layer *relu1 = new ReLU(512);
    Layer *hidden1 = new Linear(512, 128);
    Layer *relu2 = new ReLU(128);
    Layer *last = new Linear(128, 10);

    /* Add Layers to MLP Model */
    model.add(input);
    model.add(relu1);
    model.add(hidden1);
    model.add(relu2);
    model.add(last);

    /* Initialize loss Func and Trainer */
    CrossEntropyLoss loss_fn = CrossEntropyLoss();
    Trainer trainer = Trainer(model, loss_fn);

    trainer.train(train_dataset, EPOCHS, lr);


    const char* models = std::getenv("MODELS");
    if (!models) {
        std::cerr << "ERROR: MODELS env var not set!\n";
    return 0;
}
    /* Save trained model */
    std::string save_path = std::string(models) + "/model1.txt";
    model.save(save_path);

    return 0;
}