#pragma once

#include <vector>
#include <cstdint>

namespace federated {

// Byzantine fault detection using statistical methods
class ByzantineDetector {
public:
    ByzantineDetector(const std::vector<std::vector<float>>& gradients,
                     const std::vector<int>& client_ids);
    
    // Detect outlier clients using median-based approach
    std::vector<int> detectOutliers(double threshold = 2.5);
    
    // Compute pairwise distances between gradients
    std::vector<std::vector<double>> computePairwiseDistances();
    
    // Compute gradient norm for each client
    std::vector<double> computeGradientNorms();

private:
    const std::vector<std::vector<float>>& gradients_;
    const std::vector<int>& client_ids_;
    
    double computeDistance(const std::vector<float>& g1, 
                          const std::vector<float>& g2);
    double computeMedian(std::vector<double> values);
    double computeMAD(const std::vector<double>& values, double median);
};

// Gradient compression utilities
namespace compression {

// Quantize gradients to 8-bit integers
std::vector<uint8_t> quantize(const std::vector<float>& gradient,
                               float& scale, float& zero_point);

// Dequantize back to floats
std::vector<float> dequantize(const std::vector<uint8_t>& quantized,
                               float scale, float zero_point);

// Top-K sparsification
std::vector<float> topKSparsification(const std::vector<float>& gradient,
                                      int k, std::vector<int>& indices);

// Reconstruct sparse gradient
std::vector<float> reconstructSparse(const std::vector<float>& values,
                                     const std::vector<int>& indices,
                                     int total_size);

// Compute compression ratio
double computeCompressionRatio(int original_bytes, int compressed_bytes);

} // namespace compression

} // namespace federated
