/*
# 以下のパスを設定（bashで実行すればいいよ）
# export PATH=/usr/local/cuda-11.4/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64:$LD_LIBRARY_PATH
# コンパイル
# nvcc -O3 measure_dram_bandwidth.cu -o measure_dram_bandwidth
# 実行
# ./measure_dram_bandwidth
*/

#include <cuda_runtime.h>

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
    const size_t SIZE_MB = 512;
    const size_t BYTES = SIZE_MB * 1024ULL * 1024ULL;

    const int WARMUP = 5;
    const int REPEAT = 20;

    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;

    std::cout << "Allocating device memory..." << std::endl;
    std::cout << "Buffer size: " << SIZE_MB << " MB x 2" << std::endl;

    CHECK_CUDA(cudaMalloc(&d_src, BYTES));
    CHECK_CUDA(cudaMalloc(&d_dst, BYTES));

    CHECK_CUDA(cudaMemset(d_src, 1, BYTES));
    CHECK_CUDA(cudaMemset(d_dst, 0, BYTES));
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Warmup..." << std::endl;

    for (int i = 0; i < WARMUP; i++) {
        CHECK_CUDA(cudaMemcpy(d_dst, d_src, BYTES, cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<double> times_ms;
    std::vector<double> bandwidth_gbs;

    std::ofstream csv("peak_dram_bandwidth_iterations.csv");
    csv << "iteration,elapsed_ms,bandwidth_GB_per_s\n";

    std::cout << "Measurement..." << std::endl;

    for (int i = 0; i < REPEAT; i++) {
        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUDA(cudaMemcpy(d_dst, d_src, BYTES, cudaMemcpyDeviceToDevice));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float elapsed_ms_f = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms_f, start, stop));

        double elapsed_ms = static_cast<double>(elapsed_ms_f);
        double elapsed_s = elapsed_ms / 1000.0;

        double gbs = static_cast<double>(BYTES) / elapsed_s / 1e9;

        times_ms.push_back(elapsed_ms);
        bandwidth_gbs.push_back(gbs);

        csv << (i + 1) << ","
            << std::fixed << std::setprecision(6) << elapsed_ms << ","
            << std::fixed << std::setprecision(6) << gbs << "\n";

        std::cout << std::setw(3) << (i + 1)
                  << ": " << std::fixed << std::setprecision(3)
                  << elapsed_ms << " ms, "
                  << std::fixed << std::setprecision(2)
                  << gbs << " GB/s" << std::endl;
    }

    csv.close();

    double mean_ms = mean(times_ms);
    double std_ms = stddev(times_ms, mean_ms);
    double mean_gbs = mean(bandwidth_gbs);
    double std_gbs = stddev(bandwidth_gbs, mean_gbs);

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::ofstream report("peak_dram_bandwidth_result.txt");

    report << "=== Device-to-Device DRAM Bandwidth Measurement ===\n";
    report << "Date: " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "\n";
    report << "Method: cudaMemcpy DeviceToDevice\n";
    report << "Buffer size: " << SIZE_MB << " MB\n";
    report << "Bytes per iteration: " << BYTES << "\n";
    report << "Memory allocation: cudaMalloc\n";
    report << "Timing: CUDA Event\n";
    report << "Warmup: " << WARMUP << "\n";
    report << "Measurement repeats: " << REPEAT << "\n\n";

    report << "Mean elapsed time: " << std::fixed << std::setprecision(6) << mean_ms << " ms\n";
    report << "Std elapsed time: " << std::fixed << std::setprecision(6) << std_ms << " ms\n";
    report << "Mean DRAM bandwidth: " << std::fixed << std::setprecision(6) << mean_gbs << " GB/s\n";
    report << "Std DRAM bandwidth: " << std::fixed << std::setprecision(6) << std_gbs << " GB/s\n";
    report << "\nPer-iteration CSV: peak_dram_bandwidth_iterations.csv\n";

    report.close();

    std::cout << "\n==== RESULT ====\n";
    std::cout << "Mean elapsed time : " << mean_ms << " ms\n";
    std::cout << "Std elapsed time  : " << std_ms << " ms\n";
    std::cout << "Mean bandwidth    : " << mean_gbs << " GB/s\n";
    std::cout << "Std bandwidth     : " << std_gbs << " GB/s\n";
    std::cout << "Saved: peak_dram_bandwidth_result.txt\n";
    std::cout << "Saved: peak_dram_bandwidth_iterations.csv\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    return 0;
}