#include <cstdlib>
#include <vector>
#include <string>
#include "mnist_loader.hpp"
#include <iostream>

/**************************** Main ****************************/

int main(int argc, char **argv) {

    /* LOAD DATA: Call Data Loader */
    std::vector<MNISTSample> train_data = MNISTLoader::load_training(MNISTLoader::get_data_dir());


    /* Process Command Line Arguments */
    if (argc == 2 && !strcmp(argv[1], "viewhead")) {
        struct MNISTSample sample = train_data[0];
        MNISTLoader::test_print(sample);
    }

    return 0;
}

/************************** End Main **************************/
