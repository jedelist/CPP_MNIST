#pragma once
#include <string>

namespace gui {
    constexpr int GRID_SIZE = 28;
    constexpr int CELL_SIZE = 14;
    constexpr int CANVAS_SIZE = GRID_SIZE * CELL_SIZE;
    constexpr int WINDOW_WIDTH = CANVAS_SIZE;
    constexpr int WINDOW_HEIGHT = CANVAS_SIZE + 60;

    /* Clear Button */
    constexpr int BUTTON_WIDTH = 120;
    constexpr int BUTTON_HEIGHT = 40;
    constexpr int BUTTON_Y = CANVAS_SIZE + 10;

    const std::string font_path = "/System/Library/Fonts/Times.ttc";
}