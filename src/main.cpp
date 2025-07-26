#include <cstdlib>
#include <vector>
#include <string>
#include "mnist_loader.hpp"
#include <iostream>

/**************************** Main ****************************/

int main() {
    auto train_data = MNISTLoader::load_training(MNISTLoader::get_data_dir());
    auto sample = train_data[1];
    MNISTLoader::test_print(sample);

    return 0;
}

/************************** End Main **************************/
