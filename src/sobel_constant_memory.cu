#include "opencv2/opencv.hpp"
#include <cuda_runtime.h
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

__global__ void SobelEdgeKernelDivergence(unsigned char* input, unsigned char* output, int width, int height)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
    {
        output[y * width + x] = 0;
        return;
    }

    int gx =
     -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)] +
     -2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)] +
     -input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];

    int gy =
     -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)] +
      input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];

    int magnitude = abs(gx) + abs(gy);
    output[y * width + x] = (unsigned char)(magnitude > 255 ? 255 : magnitude);
}
int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cout << "Camera open failed" << endl;
        return -1;
    }

    Mat frame, gray, edges;
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    int allocatedSize = 0;
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
      
        if (size != allocatedSize)
        {
            if (d_input) cudaFree(d_input);
            if (d_output) cudaFree(d_output);
            cudaMalloc(&d_input, size);
            cudaMalloc(&d_output, size);
            allocatedSize = size;
        }

        edges.create(height, width, CV_8UC1);
        cudaMemcpy(d_input, gray.data, size, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid((width +block.x - 1) / block.x, (height + block.y - 1) / block.y);

        cudaEventRecord(start);
        SobelEdgeKernelDivergence <<< grid, block >>> (d_input, d_output, width, height);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&kernelTime, start, stop);

        cudaMemcpy(edges.data, d_output, size, cudaMemcpyDeviceToHost);
        double bandwidth = (2.0 * size) / (kernelTime / 1000.0) / 1e9;

        putText(edges, "Optimization: Minimize Control Divergence", Point(20, 30),
          FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255), 2);

        putText(edges, "Kernel Time: " + to_string(kernelTime) + " ms", Point(20, 60),
          FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255), 2);

        putText(edges, "Bandwidth: " + to_string(bandwidth) + " GB/s", Point(20, 90),
          FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255), 2);


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
