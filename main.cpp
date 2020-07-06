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
    // Open the image
    cv::Mat image = load_image("res/image.jpg");

    // Setup up the meta data for the program
    int batch_size = 1;
    int channels = 3;
    int width = image.cols;
    int height = image.rows;

    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_tensor;
    cudnnTensorDescriptor_t output_tensor;
    cudnnFilterDescriptor_t kernel;
    cudnnConvolutionDescriptor_t convolution;
    cudnnConvolutionFwdAlgoPerf_t convolution_algorithm;

    size_t workspace_bytes = 0;

    int image_bytes = batch_size * channels * width * height * sizeof(float);

    float alpha = 1.0f;
    float beta = 0.0f;
    void* d_workspace{nullptr};
    float* d_input{nullptr};
    float* d_output{nullptr};
    float* d_kernel{nullptr};
    float* h_output = new float[image_bytes];

    const float kernel_template[3][3] = {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1}
    };

    float h_kernel[3][3][3][3];
    for(int k = 0;k<3;++k) {
    for(int chan = 0;chan<3;++chan) {
        for(int r = 0;r<3;++r) {
        for(int c = 0;c<3;++c) {
            h_kernel[k][chan][r][c] = kernel_template[r][c];
        }
        }
    }
    }


    // Create and Set Descriptors
    cudnnCreate(&cudnn);

    cudnnCreateTensorDescriptor(&input_tensor);
    cudnnSetTensor4dDescriptor(input_tensor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,batch_size,channels,height,width);

    cudnnCreateTensorDescriptor(&output_tensor);
    cudnnSetTensor4dDescriptor(output_tensor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,batch_size,channels,height,width);

    cudnnCreateFilterDescriptor(&kernel);
    cudnnSetFilter4dDescriptor(kernel,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,channels,channels,channels,channels);

    cudnnCreateConvolutionDescriptor(&convolution);
    cudnnSetConvolution2dDescriptor(convolution,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);

    cudnnGetConvolutionForwardAlgorithm_v7(cudnn,input_tensor,kernel,convolution,output_tensor,CUDNN_CONVOLUTION_FWD_ALGO_GEMM,0,&convolution_algorithm);

    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_tensor, kernel, convolution, output_tensor, convolution_algorithm.algo, &workspace_bytes);    


    // Allocate memory
    cudaMalloc(&d_workspace, workspace_bytes);

    cudaMalloc(&d_input, image_bytes);
    cudaMemcpy(d_input, image.ptr<float>(0), image_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_output, image_bytes);
    cudaMemcpy(d_output, 0, image_bytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel),cudaMemcpyHostToDevice);

    // Compute the Convolution
    cudnnConvolutionForward(cudnn, &alpha, input_tensor, d_input, kernel, d_kernel, convolution, convolution_algorithm.algo, d_workspace, workspace_bytes, &beta, output_tensor, d_output);

    // Save the output image
    cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost);
    save_image("res/output.jpg", h_output, height, width);

    // Clean up the memory of the program
    delete[] h_output;
    cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_output);

    cudnnDestroyConvolutionDescriptor(convolution);
    cudnnDestroyFilterDescriptor(kernel);
    cudnnDestroyTensorDescriptor(output_tensor);
    cudnnDestroyTensorDescriptor(input_tensor);
    cudnnDestroy(cudnn);

    return 0;
}