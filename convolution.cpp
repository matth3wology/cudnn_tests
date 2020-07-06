#include <opencv2/opencv.hpp>
#include <cudnn.h>
#include <cuda_runtime.h>

cv::Mat load_image(const char* image_path) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    image.convertTo(image, CV_32FC3);
    cv::normalize(image, image, 0, 1, cv::NORM_MINMAX);
    return image;
}

void save_image(const char* output_filename, float* buffer, int height, int width) {
  cv::Mat output_image(height, width, CV_32FC3, buffer);
  cv::threshold(output_image,output_image,0,0,cv::THRESH_TOZERO);
  cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
  output_image.convertTo(output_image, CV_8UC3);
  cv::imwrite(output_filename, output_image);
}

int main() {

    cv::Mat image = load_image("res/image.jpg");

    int batch_size = 1;
    int channels = 3;
    int width = image.cols;
    int height = image.rows;

    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;
    
    int image_bytes = batch_size * channels * height * width * sizeof(float);
    size_t workspace_bytes = 0;
    void* d_workspace{nullptr};
    float* d_output{nullptr};
    float* d_input{nullptr};
    float* d_kernel{nullptr};
    const float alpha = 1, beta = 0;
    float* h_output = new float[image_bytes];
    
    const float kernel_template[3][3] = {
        {1,  1, 1},
        {1, -8, 1},
        {1,  1, 1}
    };

    float h_kernel[3][3][3][3];
    for (int kernel = 0; kernel < 3; ++kernel) {
    for (int channel = 0; channel < 3; ++channel) {
        for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            h_kernel[kernel][channel][row][column] = kernel_template[row][column];
        }
        }
    }
    }

    // Create and Set the descriptors
    cudnnCreate(&cudnn);

    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,3,image.rows,image.cols);


    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,3,image.rows,image.cols);

    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,3,3,3,3);

    cudnnCreateConvolutionDescriptor(&convolution_descriptor);
    cudnnSetConvolution2dDescriptor(convolution_descriptor,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);

    cudnnGetConvolutionForwardAlgorithm_v7(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor,output_descriptor,CUDNN_CONVOLUTION_FWD_ALGO_GEMM,0,&convolution_algorithm);

    cudnnGetConvolutionForwardWorkspaceSize(cudnn,input_descriptor,kernel_descriptor,convolution_descriptor,output_descriptor,convolution_algorithm.algo,&workspace_bytes);


    // Allocating Memory
    cudaMalloc(&d_workspace, workspace_bytes);


    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);


    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    // Convolution
    cudnnConvolutionForward(cudnn,&alpha,input_descriptor,d_input,kernel_descriptor,d_kernel,convolution_descriptor,convolution_algorithm.algo,d_workspace,workspace_bytes,&beta,output_descriptor,d_output);

    // Save the image
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    save_image("res/cudnn-out.jpg", h_output, height, width);

    // Free up memory
    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);
}