#include "byzantine_detector.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>

namespace federated {

ByzantineDetector::ByzantineDetector(
    const std::vector<std::vector<float>>& gradients,
    const std::vector<int>& client_ids)
    : gradients_(gradients), client_ids_(client_ids) {}

std::vector<int> ByzantineDetector::detectOutliers(double threshold) {
    std::vector<int> outliers;
    
    if (gradients_.size() < 3) {
        // Need at least 3 clients for meaningful outlier detection
        return outliers;
    }
    
    // Compute gradient norms
    std::vector<double> norms = computeGradientNorms();
    
    // Compute median and MAD (Median Absolute Deviation)
    double median = computeMedian(norms);
    double mad = computeMAD(norms, median);
    
    std::cout << "Median norm: " << median << ", MAD: " << mad << std::endl;
    
    // Detect outliers using modified Z-score
    // A gradient is an outlier if |norm - median| / MAD > threshold
    for (size_t i = 0; i < norms.size(); ++i) {
        double modified_z_score = std::abs(norms[i] - median) / (mad + 1e-10);
        
        if (modified_z_score > threshold) {
            std::cout << "Client " << client_ids_[i] << " detected as Byzantine "
                     << "(norm=" << norms[i] << ", z-score=" << modified_z_score << ")" 
                     << std::endl;
            outliers.push_back(client_ids_[i]);
        }
    }
    
    return outliers;
}

std::vector<std::vector<double>> ByzantineDetector::computePairwiseDistances() {
    size_t n = gradients_.size();
    std::vector<std::vector<double>> distances(n, std::vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double dist = computeDistance(gradients_[i], gradients_[j]);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }
    
    return distances;
}

std::vector<double> ByzantineDetector::computeGradientNorms() {
    std::vector<double> norms;
    norms.reserve(gradients_.size());
    
    for (const auto& gradient : gradients_) {
        double sum_squares = 0.0;
        for (float val : gradient) {
            sum_squares += val * val;
        }
        norms.push_back(std::sqrt(sum_squares));
    }
    
    return norms;
}

double ByzantineDetector::computeDistance(const std::vector<float>& g1,
                                         const std::vector<float>& g2) {
    double sum_squares = 0.0;
    
    for (size_t i = 0; i < g1.size(); ++i) {
        double diff = g1[i] - g2[i];
        sum_squares += diff * diff;
    }
    
    return std::sqrt(sum_squares);
}

double ByzantineDetector::computeMedian(std::vector<double> values) {
    if (values.empty()) return 0.0;
    
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) / 2.0;
    } else {
        return values[mid];
    }
}

double ByzantineDetector::computeMAD(const std::vector<double>& values, double median) {
    std::vector<double> abs_deviations;
    abs_deviations.reserve(values.size());
    
    for (double val : values) {
        abs_deviations.push_back(std::abs(val - median));
    }
    
    return computeMedian(abs_deviations);
}

namespace compression {

std::vector<uint8_t> quantize(const std::vector<float>& gradient, 
                               float& scale, float& zero_point) {
    // Find min and max values
    float min_val = *std::min_element(gradient.begin(), gradient.end());
    float max_val = *std::max_element(gradient.begin(), gradient.end());
    
    // Compute scale and zero point
    scale = (max_val - min_val) / 255.0f;
    zero_point = -min_val / scale;
    
    // Quantize
    std::vector<uint8_t> quantized(gradient.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < gradient.size(); ++i) {
        float scaled = gradient[i] / scale + zero_point;
        quantized[i] = static_cast<uint8_t>(
            std::min(255.0f, std::max(0.0f, std::round(scaled)))
        );
    }
    
    return quantized;
}

std::vector<float> dequantize(const std::vector<uint8_t>& quantized,
                               float scale, float zero_point) {
    std::vector<float> gradient(quantized.size());
    
    #pragma omp parallel for
    for (size_t i = 0; i < quantized.size(); ++i) {
        gradient[i] = (static_cast<float>(quantized[i]) - zero_point) * scale;
    }
    
    return gradient;
}

std::vector<float> topKSparsification(const std::vector<float>& gradient,
                                      int k, std::vector<int>& indices) {
    // Create pairs of (absolute value, index)
    std::vector<std::pair<float, int>> abs_values;
    abs_values.reserve(gradient.size());
    
    for (size_t i = 0; i < gradient.size(); ++i) {
        abs_values.push_back({std::abs(gradient[i]), static_cast<int>(i)});
    }
    
    // Partial sort to get top-k
    std::partial_sort(abs_values.begin(), 
                     abs_values.begin() + k,
                     abs_values.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract top-k values and indices
    std::vector<float> values;
    values.reserve(k);
    indices.clear();
    indices.reserve(k);
    
    for (int i = 0; i < k; ++i) {
        int idx = abs_values[i].second;
        indices.push_back(idx);
        values.push_back(gradient[idx]);
    }
    
    return values;
}

std::vector<float> reconstructSparse(const std::vector<float>& values,
                                     const std::vector<int>& indices,
                                     int total_size) {
    std::vector<float> gradient(total_size, 0.0f);
    
    for (size_t i = 0; i < values.size(); ++i) {
        gradient[indices[i]] = values[i];
    }
    
    return gradient;
}

double computeCompressionRatio(int original_bytes, int compressed_bytes) {
    return 1.0 - (static_cast<double>(compressed_bytes) / original_bytes);
}

} // namespace compression
} // namespace federated