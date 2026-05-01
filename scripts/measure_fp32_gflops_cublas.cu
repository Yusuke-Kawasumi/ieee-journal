/*
# 実行上の注意点
# コンパイル時パスを設定したのちにコンパイル
# export PATH=/usr/local/cuda-11.4/bin:$PATH
# nvcc -O3 measure_fp32_gflops_cublas.cu -lcublas -o measure_fp32_gflops_cublas
# もう一度パスの設定をして実行
# export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
# ./measure_fp32_gflops_cublas
*/

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t stat = (call); \
    if (stat != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << stat \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

static double mean(const std::vector<double>& x) {
    double s = 0.0;
    for (double v : x) s += v;
    return s / x.size();
}

static double stddev(const std::vector<double>& x, double m) {
    if (x.size() < 2) return 0.0;
    double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return std::sqrt(s / (x.size() - 1));
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const int WARMUP = 10;
    const int REPEAT = 100;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    const double flops = 2.0 * M * N * K;

    std::cout << "Allocating host memory..." << std::endl;

    std::vector<float> h_A(sizeA);
    std::vector<float> h_B(sizeB);
    std::vector<float> h_C(sizeC);

    for (size_t i = 0; i < sizeA; i++) h_A[i] = 1.0f;
    for (size_t i = 0; i < sizeB; i++) h_B[i] = 1.0f;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    std::cout << "Allocating device memory..." << std::endl;

    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(float)));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 念のため Tensor Core / TF32 系ではなく通常FP32に寄せる
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Warmup..." << std::endl;

    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M
        ));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> times_ms;
    std::vector<double> gflops_values;

    std::ofstream csv("peak_fp32_cublas_iterations.csv");
    csv << "iteration,elapsed_ms,gflops\n";

    std::cout << "Measurement..." << std::endl;

    for (int i = 0; i < REPEAT; i++) {
        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            d_A, M,
            d_B, K,
            &beta,
            d_C, M
        ));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms_f = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms_f, start, stop));

        double elapsed_ms = static_cast<double>(elapsed_ms_f);
        double elapsed_s = elapsed_ms / 1000.0;
        double gflops = flops / elapsed_s / 1e9;

        times_ms.push_back(elapsed_ms);
        gflops_values.push_back(gflops);

        csv << (i + 1) << ","
            << std::fixed << std::setprecision(6) << elapsed_ms << ","
            << std::fixed << std::setprecision(6) << gflops << "\n";

        std::cout << std::setw(3) << (i + 1)
                  << ": " << std::fixed << std::setprecision(3)
                  << elapsed_ms << " ms, "
                  << std::fixed << std::setprecision(2)
                  << gflops << " GFLOPS" << std::endl;
    }

    csv.close();

    double mean_gflops = mean(gflops_values);
    double std_gflops = stddev(gflops_values, mean_gflops);
    double mean_ms = mean(times_ms);
    double std_ms = stddev(times_ms, mean_ms);

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::ofstream report("peak_fp32_cublas_result.txt");

    report << "=== cuBLAS FP32 SGEMM Peak Measurement ===\n";
    report << "Date: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "\n";
    report << "Method: cuBLAS cublasSgemm\n";
    report << "Matrix size: M=" << M << ", N=" << N << ", K=" << K << "\n";
    report << "Precision: FP32\n";
    report << "Math mode: CUBLAS_DEFAULT_MATH\n";
    report << "Warmup: " << WARMUP << "\n";
    report << "Measurement repeats: " << REPEAT << "\n\n";
    report << "FLOPs per iteration: " << std::fixed << std::setprecision(0) << flops << "\n";
    report << "Mean elapsed time: " << std::fixed << std::setprecision(6) << mean_ms << " ms\n";
    report << "Std elapsed time: " << std::fixed << std::setprecision(6) << std_ms << " ms\n";
    report << "Mean FP32 performance: " << std::fixed << std::setprecision(6) << mean_gflops << " GFLOPS\n";
    report << "Std FP32 performance: " << std::fixed << std::setprecision(6) << std_gflops << " GFLOPS\n";
    report << "\nPer-iteration CSV: peak_fp32_cublas_iterations.csv\n";

    report.close();

    std::cout << "\n==== RESULT ====\n";
    std::cout << "Mean elapsed time: " << mean_ms << " ms\n";
    std::cout << "Std elapsed time : " << std_ms << " ms\n";
    std::cout << "Mean GFLOPS      : " << mean_gflops << "\n";
    std::cout << "Std GFLOPS       : " << std_gflops << "\n";
    std::cout << "Saved: peak_fp32_cublas_result.txt\n";
    std::cout << "Saved: peak_fp32_cublas_iterations.csv\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUBLAS(cublasDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}