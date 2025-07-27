#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <cstdlib>

typedef float data_t;

/* Base Layer Class. Linear adn ReLU will be derived */
class Layer {
    protected:
        int inSize;
        int outSize;

    public:
        Layer (int in, int out) : inSize(in), outSize(out) {}

        /* Forward: Computes the forward pass of layer */
        virtual std::vector<data_t> forward(const std::vector<data_t> &input) = 0;

        /* Backward: Takes grad_output (dL/dy), returns grad_input (dL/dx) */
        virtual std::vector<data_t> backward(const std::vector<data_t> &grad_output) = 0;

        /* Updates the parameters (only required for learnable layers */
        virtual void update(float lr) {}

        virtual ~Layer() = default;

        /* Getters */
        int input_size() const { return inSize; }
        int output_size() const { return outSize; }
        virtual void load(std::ifstream& in) = 0;
        virtual void save(std::ofstream& out) = 0;
};

class Linear : public Layer {
    public:

        Eigen::MatrixXf W;                      /* Size = [outSize × inSize] */
        Eigen::VectorXf b;                      /* Size = [outSize] */
        Eigen::MatrixXf grad_W;                 /* Size = [outSize × inSize] */
        Eigen::VectorXf grad_b;                 /* size = [outSize] */
        Eigen::VectorXf last_input;             /* size = [inSize] Store for backward */

        Linear (int in_features, int out_features);
        std::vector<data_t> forward(const std::vector<data_t> &input) override; 
        std::vector<data_t> backward(const std::vector<data_t> &grad_output) override; 
        void update(float lr) override;
        void load(std::ifstream& in) override;
        void save(std::ofstream& out) override;
};

class ReLU : public Layer {
    public:
        Eigen::VectorXf mask;

        ReLU(int features);
        std::vector<data_t> forward(const std::vector<data_t> &input) override;
        std::vector<data_t> backward(const std::vector<data_t> &grad_output) override; /* dReLU*/
        void load(std::ifstream& in) override {}  // No params to load
        void save(std::ofstream& out) override {}
};

/* Cross Entropy Loss declared. Did not have it as a layer */
class CrossEntropyLoss {
    private:
        std::vector<data_t> last_softmax;

    public:
        data_t forward(const std::vector<data_t> &softmax_output, uint8_t label);
        static std::vector<data_t> softmax(const std::vector<data_t> &input);
        std::vector<data_t> backward(uint8_t label);
};