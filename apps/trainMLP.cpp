#include <iostream>
#include <vector>
#include <string>
#include "MLP.hpp"
#include "layers.hpp"
#include "mnist_loader.hpp"
#include "trainer.hpp"
#include "train_config.hpp"

typedef float data_t;

int main (int argc, char **argv) {

    /* Load training dataset */
    std::vector<MNISTSample> train_dataset = MNISTLoader::load_training(MNISTLoader::get_data_dir());

    /* Initialize MLP model */
    MLP model = MLP(config::LR);

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

    std::cout << "Training Model " << config::MODEL_NUM << 
    " which will be saved to $MODELS/model" << config::MODEL_NUM << ".txt" << std::endl;
    trainer.train(train_dataset, config::EPOCHS, config::LR);

    const char* models_dir = std::getenv("MODELS");
    if (!models_dir) {
        std::cerr << "ERROR: MODELS env var not set!\n Ensure you `source job_bashrc`. \n";
        return 0;
    }

    /* Save trained model */
    //const char* MODEL_NUM = argv[1];
    std::string save_path = std::string(models_dir) + "/model" + std::to_string(config::MODEL_NUM) + ".txt";

    model.save(save_path);

    return 0;
}