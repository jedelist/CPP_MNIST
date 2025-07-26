#include <cstdlib>
#include <string>
#include "mnist_loader.hpp"
#include <iostream>

/* Prototypes */
std::string get_data_dir();

/**************************** Main ****************************/

int main() {
    auto train_data = MNISTLoader::load_training(get_data_dir());
}
/************************** End Main **************************/


std::string get_data_dir() {
    const char* env_dir = std::getenv("DATA");
    std::cout << env_dir << std::endl;
    if (env_dir) return std::string(env_dir);
    return "data";
}