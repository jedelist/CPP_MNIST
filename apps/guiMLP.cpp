#include <iostream>
#include <cmath>
#include <string>
#include <SFML/Graphics.hpp>
#include "MLP.hpp"
#include "layers.hpp"
#include "trainer.hpp"
#include "gui_config.hpp"
#include "gui_helpers.hpp"
#include "train_config.hpp"

int main(int argc, char ** argv) {
    sf::RenderWindow window(sf::VideoMode(gui::WINDOW_WIDTH, gui::WINDOW_HEIGHT), "Draw a Digit (28x28)");

    /* Create 28x28 grid of false pixels */
    std::vector<std::vector<bool>> grid(gui::GRID_SIZE, std::vector<bool>(gui::GRID_SIZE, false));

    /* Load model/weights */
    const char* models_dir = std::getenv("MODELS");
    const char* MODEL_NUM = argv[1];
    std::string loaded_model = std::string(models_dir) + "/model" + std::string(MODEL_NUM) + ".txt";
    float lr = 0.0001f;
    MLP model(lr);
    model.add(new Linear(784, 512));
    model.add(new ReLU(512));
    model.add(new Linear(512, 128));
    model.add(new ReLU(128));
    model.add(new Linear(128, 10));
    model.load(loaded_model);

    sf::Font font;
    if (!font.loadFromFile(gui::font_path)) {
        std::cerr << "Failed to load font from " << gui::font_path << std::endl;
        return 1;
    }

    sf::Text predict_text("Predict", font, 24);
    predict_text.setFillColor(sf::Color::Black);
    predict_text.setPosition(gui::WINDOW_WIDTH / 2 - 40, gui::CANVAS_SIZE + 10);

    /* Predict button settings */
    sf::RectangleShape predict_button(sf::Vector2f(100, 40));
    predict_button.setFillColor(sf::Color(200, 200, 200));
    predict_button.setOutlineThickness(2);
    predict_button.setOutlineColor(sf::Color::Black);
    predict_button.setPosition(gui::WINDOW_WIDTH / 2 - 50, gui::CANVAS_SIZE + 5);

    /* Prediction text for model output */
    sf::Text result_text("", font, 24);
    result_text.setFillColor(sf::Color::Black);
    result_text.setPosition(10, gui::CANVAS_SIZE + 10);

    /* Clear button settings */
    sf::RectangleShape clearButton(sf::Vector2f(gui::BUTTON_WIDTH, gui::BUTTON_HEIGHT));
    clearButton.setPosition(gui::CANVAS_SIZE - gui::BUTTON_WIDTH - 20, gui::BUTTON_Y);  // Right side
    clearButton.setFillColor(sf::Color(200, 100, 100));  // Red-ish

    sf::Text clearText;
    clearText.setFont(font);
    clearText.setString("Clear");
    clearText.setCharacterSize(20);
    clearText.setFillColor(sf::Color::White);
    clearText.setPosition(clearButton.getPosition().x + 25, clearButton.getPosition().y + 7);

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();

            /* Drawing on canvas clicking pixels */
            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
                int x = mouse_pos.x / gui::CELL_SIZE;
                int y = mouse_pos.y / gui::CELL_SIZE;
                if (x >= 0 && x < gui::GRID_SIZE && y >= 0 && y < gui::GRID_SIZE) {
                    grid[y][x] = true;
                }
            }

            if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
                int mx = event.mouseButton.x;
                int my = event.mouseButton.y;
                /* Predict with model with the "Predict" bytton */
                if (predict_button.getGlobalBounds().contains(mx, my)) {
                    std::vector<float> input = gridToInput(grid);
                    int prediction = model.predict(input);
                    result_text.setString("Prediction: " + std::to_string(prediction));
                }
                /* Clear grid for clear button */
                if (clearButton.getGlobalBounds().contains(mx, my)) {
                    result_text.setString("Prediction: ");
                    for (int y = 0; y < gui::GRID_SIZE; ++y)
                        for (int x = 0; x < gui::GRID_SIZE; ++x)
                            grid[y][x] = false;
                }
            }
        }

        window.clear(sf::Color::White);

        /* Draw grid */
        for (int y = 0; y < gui::GRID_SIZE; ++y) {
            for (int x = 0; x < gui::GRID_SIZE; ++x) {
                sf::RectangleShape cell(sf::Vector2f(gui::CELL_SIZE - 1, gui::CELL_SIZE - 1));
                cell.setPosition(x * gui::CELL_SIZE, y * gui::CELL_SIZE);
                cell.setFillColor(grid[y][x] ? sf::Color::Black : sf::Color::White);
                window.draw(cell);
            }
        }

        /* Draw buttons and text */
        window.draw(predict_button);
        window.draw(predict_text);
        window.draw(result_text);
        window.draw(clearButton);
        window.draw(clearText);

        window.display();
    }

    return 0;
}
