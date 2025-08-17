# **CPP_MNIST**

Personal project where I created a Multi-layer Perceptron (MLP) from scratch in C++ to gain a better under-the-hood understanding of neural networks and deep learning frameworks' implementation. I used CMake for this project, so it should build properly on all systems and compatible with any compiler. This project is extrememly modular to support future expansion and C++ object-oriented best practices. I plan to add better support for hyperparameter tuning, more optimization features during training, and more types of layers (such as 2D convolution). Follow the instructions below to use this software :).

- **No external ML libraries** — all math, backpropagation, and layers are hand-coded  
- **Fully modular** — supports flexible architecture and easy layer expansion for future work
- **Model serialization** — train, save, and reload models from disk  
- **GUI integration** — draw digits and classify them using trained models in a GUI

## **1. Setup and Build**
First, source the bash profile to set environment variables and aliases:

```bash
source job_bashrc
```

To perform a full Build, run from project root:

```bash
bld -B
```

To Build only changed files, run from project root:

```bash
bld
```

- Ensure the executables were properly built. They should be located in `/bin`. 

## **2. Train a Model:**

To train MLP, run the following executable: 

```bash
./bin/trainMLP
```
- **Important:** The model number used for saving must be specified in train_config.hpp (located in /include).
Example: Setting constexpr int MODEL_NUM = 3; will save the trained model as models/model3.txt. This is important for testing/evaluating the model later, and using it for inference when using the GUI.

- Inside `/apps/trainMLP.cpp` you can configure the model's architecture and layer structure to your liking. Also, set the hyperparameters for training, such as number of epochs and learning rate in `/include/trian_config.hpp`. 

## **3. Evaluate the Model's Performance:**

To evaluate the model's performance, run inference on the MNIST test dataset by running the following:

```bash
./bin/testMLP <MODEL_NUM>
```
- **Important:** The "MODEL_NUM" above is a placeholder for the model number that your saved model is saved as inside `/models`. For example, if you want to load `model3.txt` for testMLP, you will run `./bin/testMLP 3`.

- Note: if you changed the model's architecture or layer structure from the default during training, ensure the architecture initialization in `/apps/testMLP.cpp` matches the architecture/structure you used in training before you build and attempt to run the executable.

## **4. Run the GUI and Predict:**

To run inference with a trained model using the GUI, run the following:

```bash
./bin/guiMLP <MODEL_NUM>
```

- **Important:** The "MODEL_NUM" above is a placeholder that for the model number that your saved model is named as inside `/models`. For example, if you want to load `model3.txt` for guiMLP, you will run `./bin/guiMLP 3`.

- The GUI lets you draw digits in a 28x28 grid and submit them to the classifier for prediction.

- Note: if you changed the model's architecture or layer structure from the default during training, ensure the architecture initialization in `/apps/guiMLP.cpp` matches the architecture/structure you used in training before you build and attempt to run the executable.

## **Project Structure:**

```bash
CPP_MNIST/
├── bin/              # Compiled binaries/executables
├── include/          # Header files (configs, layers, MLP, etc.)
├── src/              # Core source code
├── models/           # Saved models (ex., model0.txt, model1.txt)
├── apps/             # Contains source main()s for trainMLP, testMLP and guiMLP
├── data/             # Raw MNIST data
├── build/            # CMake build configuration files (ignore)
├── scripts/          # Helper and build scripts
├── job_bashrc        # Helper script for environment varaibles and aliases
├── CMakeLists.txt    # Build configuration
└── README.md
```

## **Future Work and Enhancements:**

- Hyperparameter tuning support (dynamic learning rate scheduler, batch size, etc.)

- Optimizers beyond vanilla gradient descent (momentum, Adam, SGD)

- Support for new layers (2D Convolution, Max Pooling)

- Live training visualization or plotting

- Enhanced error reporting and logging