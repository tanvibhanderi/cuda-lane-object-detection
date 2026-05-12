#include "opencv2/opencv.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

__global__ void SobelEdgeKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int gx =
          -1 * input[(y - 1) * width + (x - 1)] + 0 * input[(y - 1) * width + x] + 1 * input[(y - 1) * width + (x + 1)] +
          -2 * input[y * width + (x - 1)] + 0 * input[y * width + x] + 2 * input[y * width + (x + 1)] +
          -1 * input[(y + 1) * width + (x - 1)] + 0 * input[(y + 1) * width + x] + 1 * input[(y + 1) * width + (x + 1)];

        int gy =
          -1 * input[(y - 1) * width + (x - 1)] + -2 * input[(y - 1) * width + x] + -1 * input[(y - 1) * width + (x + 1)] +
           0 * input[y * width + (x - 1)] + 0 * input[y * width + x] + 0 * input[y * width + (x + 1)] +
           1 * input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + 1 * input[(y + 1) * width + (x + 1)];

        int magnitude = abs(gx) + abs(gy);

        if (magnitude > 255)
            magnitude = 255;

        output[y * width + x] = (unsigned char)magnitude;
    }
    else if (x < width && y < height)
    {
        output[y * width + x] = 0;
    }
}

int main()
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "Camera open failed" << endl;
        return -1;
    }

    Mat frame;
    Mat gray;
    Mat edges;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    float kernelTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    while (true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        int width = gray.cols;
        int height = gray.rows;
        int size = width * height * sizeof(unsigned char);

        if (!d_input)
        {
            cudaMalloc(&d_input, size);
            cudaMalloc(&d_output, size);
        }

        edges.create(height, width, CV_8UC1);

        cudaMemcpy(d_input, gray.data, size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width +15) / 16, (height + 15) / 16);

        cudaEventRecord(start);

        SobelEdgeKernel <<< grid, block >>> (d_input, d_output, width, height);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelTime, start, stop);

        cudaMemcpy(edges.data, d_output, size, cudaMemcpyDeviceToHost);

        double bandwidth = (2.0 * size) / (kernelTime / 1000.0) / 1e9;

        putText(edges, "Kernel Time: " + to_string(kernelTime) + " ms", Point(20, 30),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255), 2);

        putText(edges, "Bandwidth: " + to_string(bandwidth) + " GB/s", Point(20, 60),
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255), 2);

        imshow("Original Video", frame);
        imshow("Lane Edge Detection", edges);

        if (waitKey(1) == 27)
            break;
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
