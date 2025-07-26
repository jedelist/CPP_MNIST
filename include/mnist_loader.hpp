#pragma once
#include <vector>
#include <cstdint>
#include <string>

#define ROW_LEN 28

/* Stores an the pair [image, label] or [x,y] */
struct MNISTSample {
    std::vector<float> pixels;
    uint8_t label;
};

class MNISTLoader {
public:
    static std::vector<MNISTSample> load_training(const std::string &data_dir);     /* Needs to have length N (training set) */
    static std::vector<MNISTSample> load_test(const std::string &data_dir);         /* Needs to have length T (testing set)  */
    static std::string get_data_dir();
    static void test_print(struct MNISTSample sample);
};
