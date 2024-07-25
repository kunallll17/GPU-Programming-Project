#include <cuda_runtime.h>
#include "image_processing.h"

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, const float* filter, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelIndex = (row * width + col) * channels;
        float newValue = 0.0f;

        for (int filterRow = 0; filterRow < filterWidth; ++filterRow) {
            for (int filterCol = 0; filterCol < filterWidth; ++filterCol) {
                int imageRow = row + filterRow - filterWidth / 2;
                int imageCol = col + filterCol - filterWidth / 2;

                if (imageRow >= 0 && imageRow < height && imageCol >= 0 && imageCol < width) {
                    int imageIndex = (imageRow * width + imageCol) * channels;
                    newValue += input[imageIndex] * filter[filterRow * filterWidth + filterCol];
                }
            }
        }
        output[pixelIndex] = static_cast<unsigned char>(newValue);
    }
}

void applyGaussianBlur(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int imageSize = width * height * channels;
    unsigned char* d_input;
    unsigned char* d_output;
    float h_filter[] = {1/16.0f, 2/16.0f, 1/16.0f, 2/16.0f, 4/16.0f, 2/16.0f, 1/16.0f, 2/16.0f, 1/16.0f};
    float* d_filter;

    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_filter, sizeof(h_filter));

    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(h_filter), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels, d_filter, 3);

    cudaMemcpy(output, d_output, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
}
