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

#define MODEL_NUM 1

int main(){
    float lr = 0.0001f;
    int correct_count = 0;
    int wrong_count = 0;
    int total_count;
    float accuracy;

    const char* models_dir = std::getenv("MODELS");

    std::string loaded_model = std::string(models_dir) + "/model" + std::to_string(MODEL_NUM) + ".txt";
    MLP model(lr);

    /* Match the model architecture to the model that was saved: */
    model.add(new Linear(784, 512));
    model.add(new ReLU(512));
    model.add(new Linear(512, 128));
    model.add(new ReLU(128));
    model.add(new Linear(128, 10));
    std::cout << loaded_model << std::endl;
    model.load(loaded_model);

    std::vector<MNISTSample> test_dataset = MNISTLoader::load_test(MNISTLoader::get_data_dir());
    total_count = test_dataset.size();

    for (int i = 0; i < test_dataset.size(); i++) {
        int result = model.predict(test_dataset[i].pixels);
        if (result == static_cast<int>(test_dataset[i].label)) {
            correct_count++;
        } else {
            wrong_count++;
        }
    }

    accuracy = static_cast<float>(correct_count) / total_count;

    std::cout << std::endl << "==================== TEST RESULTS ====================" << std::endl << std::endl;
    std::cout << "Model " << MODEL_NUM << " accuracy is: " << accuracy << std::endl;
    std::cout << "Correct count: " << correct_count << std::endl;
    std::cout << "Wrong count: " << wrong_count << std::endl << std::endl;
    std::cout << "==================== END RESULTS ====================" << std::endl << std::endl;
    
    return 0;
}