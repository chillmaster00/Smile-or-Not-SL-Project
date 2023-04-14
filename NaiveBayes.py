import numpy as np

"""
Plan:
1. Get data (already implemented)
2. Initialize arrays for holding counts of each color-intensity
    0. Number of features n should be 12288 for 64*64(pixels)*3(RGB/pixel)
    1. For each pixel-color, need to hold 256 variables, so shape is (12288, 256)
        1. Index by (pixel-color, intensity-level)
        2. Call it intensity_counts?
    2. For each feature, have total number
        1. Index by (pixel-color)
        2. Call it total_counts?
3. Process data into the count arrays
4. For testing, throw into naive bayes using m-intensity
"""