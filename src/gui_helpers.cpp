#include "gui_helpers.hpp"
#include "gui_config.hpp"

std::vector<float> gridToInput(const std::vector<std::vector<bool>>& grid) {
    std::vector<float> input;
    for (int y = 0; y < gui::GRID_SIZE; ++y) {
        for (int x = 0; x < gui::GRID_SIZE; ++x) {
            input.push_back(grid[y][x] ? 1.0f : 0.0f);
        }
    }
    return input;
}