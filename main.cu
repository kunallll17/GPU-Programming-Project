#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "image_processing.h"

void processImages(const std::string& directory) {
    std::vector<std::string> imageFiles = {
        // Add the image file names or use a file system library to iterate over the directory
    };

    for (const auto& filename : imageFiles) {
        cv::Mat image = cv::imread(directory + filename, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error loading image: " << filename << std::endl;
            continue;
        }

        cv::Mat outputImage(image.size(), image.type());
        applyGaussianBlur(image.ptr<unsigned char>(), outputImage.ptr<unsigned char>(), image.cols, image.rows, image.channels());

        cv::imwrite("output/" + filename, outputImage);
    }
}

int main() {
    std::string imageDirectory = "data/small_images/";
    processImages(imageDirectory);

    return 0;
}
