#include "mnist_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

/* This is private function, only used in this file */
static uint32_t read_uint32(std::ifstream &f) {
    unsigned char bytes[4];
    f.read(reinterpret_cast<char*>(bytes), 4);
    if (!f) throw std::runtime_error("Unexpected EOF while reading uint32");
    return (uint32_t(bytes[0]) << 24) |
           (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8)  |
            uint32_t(bytes[3]);
}

/* Private loader to this file */
static std::vector<MNISTSample> load_pair(const std::string &image_path, const std::string &label_path) {
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);
    if (!image_file) throw std::runtime_error("Could not open " + image_path);
    if (!label_file) throw std::runtime_error("Could not open " + label_path);

    uint32_t magic_img = read_uint32(image_file);
    if (magic_img != 2051) throw std::runtime_error("Invalid image magic number!");
    uint32_t num_images = read_uint32(image_file);
    uint32_t rows = read_uint32(image_file);
    uint32_t cols = read_uint32(image_file);

    uint32_t magic_lbl = read_uint32(label_file);
    if (magic_lbl != 2049) throw std::runtime_error("Invalid label magic number!");
    uint32_t num_labels = read_uint32(label_file);
    if (num_images != num_labels) throw std::runtime_error("Image/label count mismatch!");

    std::cout << "Loading " << num_images << " samples (" << rows << "x" << cols << ")\n";

    std::vector<MNISTSample> dataset;
    dataset.reserve(num_images);
    const uint32_t img_size = rows * cols;

    for (uint32_t i = 0; i < num_images; i++) {
        MNISTSample sample;
        sample.pixels.resize(img_size);
        for (uint32_t j = 0; j < img_size; j++) {
            unsigned char pixel;
            image_file.read(reinterpret_cast<char*>(&pixel), 1);
            sample.pixels[j] = static_cast<float>(pixel) / 255.0f;
        }
        unsigned char lbl;
        label_file.read(reinterpret_cast<char*>(&lbl), 1);
        sample.label = lbl;
        dataset.push_back(std::move(sample));
    }

    return dataset;
}

/* Public project-wide APIs for loading MNIST training and test data */
std::vector<MNISTSample> MNISTLoader::load_training(const std::string &data_dir) {
    return load_pair(
        data_dir + "/train-images-idx3-ubyte",
        data_dir + "/train-labels-idx1-ubyte"
    );
}

std::vector<MNISTSample> MNISTLoader::load_test(const std::string &data_dir) {
    return load_pair(
        data_dir + "/t10k-images-idx3-ubyte",
        data_dir + "/t10k-labels-idx1-ubyte"
    );
}

std::string MNISTLoader::get_data_dir() {
    const char* env_dir = std::getenv("DATA");
    std::cout << "Data Directory Path: " << env_dir << std::endl;
    if (env_dir) return std::string(env_dir);
    return "data";
}

void MNISTLoader::test_print(struct MNISTSample sample) {
    const std::vector<float> image = sample.pixels;

    for (int i = 0; i < ROW_LEN; i++) {
        for (int j = 0; j < ROW_LEN; j++) {
            float val = image[i * ROW_LEN + j] < 0.5f ? 0.0f : 1.0f;
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}