#include "aggregator.h"
#include "byzantine_detector.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>
#include <iostream>

namespace federated {

GradientAggregator::GradientAggregator(int num_parameters, int expected_clients)
    : num_parameters_(num_parameters),
      expected_clients_(expected_clients) {
    gradients_.reserve(expected_clients);
    client_ids_.reserve(expected_clients);
}

void GradientAggregator::addGradient(const std::vector<float>& gradient, int client_id) {
    if (gradient.size() != static_cast<size_t>(num_parameters_)) {
        std::cerr << "Warning: Gradient size mismatch. Expected " 
                  << num_parameters_ << ", got " << gradient.size() << std::endl;
        return;
    }
    
    gradients_.push_back(gradient);
    client_ids_.push_back(client_id);
    
    // Compute and store gradient norm
    double norm = computeGradientNorm(gradient);
    stats_.client_gradient_norms.push_back(norm);
}

std::vector<float> GradientAggregator::aggregate() {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (gradients_.empty()) {
        std::cerr << "Error: No gradients to aggregate" << std::endl;
        return std::vector<float>(num_parameters_, 0.0f);
    }
    
    std::cout << "Aggregating " << gradients_.size() << " gradients..." << std::endl;
    
    // Byzantine fault tolerance: Use median-based aggregation
    ByzantineDetector detector(gradients_, client_ids_);
    byzantine_clients_ = detector.detectOutliers();
    
    std::cout << "Detected " << byzantine_clients_.size() << " Byzantine clients" << std::endl;
    
    // Filter out Byzantine gradients
    std::vector<std::vector<float>> filtered_gradients;
    for (size_t i = 0; i < gradients_.size(); ++i) {
        if (std::find(byzantine_clients_.begin(), byzantine_clients_.end(), 
                     client_ids_[i]) == byzantine_clients_.end()) {
            filtered_gradients.push_back(gradients_[i]);
        }
    }
    
    // Compute median gradient (coordinate-wise median)
    std::vector<float> aggregated = computeMedianGradient();
    
    // Compute statistics
    auto end = std::chrono::high_resolution_clock::now();
    stats_.aggregation_time_ms = 
        std::chrono::duration<double, std::milli>(end - start).count();
    stats_.outliers_detected = byzantine_clients_.size();
    
    // Compute median gradient norm
    std::vector<double> norms = stats_.client_gradient_norms;
    std::sort(norms.begin(), norms.end());
    stats_.median_gradient_norm = norms[norms.size() / 2];
    
    // Estimate compression ratio (simplified)
    stats_.compression_ratio = 0.75; // Placeholder
    
    std::cout << "Aggregation complete in " << stats_.aggregation_time_ms 
              << "ms" << std::endl;
    
    return aggregated;
}

std::vector<float> GradientAggregator::computeMedianGradient() {
    if (gradients_.empty()) {
        return std::vector<float>(num_parameters_, 0.0f);
    }
    
    std::vector<float> median_gradient(num_parameters_);
    
    // For each parameter, compute median across all clients
    #pragma omp parallel for
    for (int param_idx = 0; param_idx < num_parameters_; ++param_idx) {
        std::vector<float> values;
        values.reserve(gradients_.size());
        
        // Collect values for this parameter from all gradients
        for (const auto& gradient : gradients_) {
            // Skip Byzantine clients
            bool is_byzantine = false;
            for (size_t i = 0; i < gradients_.size(); ++i) {
                if (&gradient == &gradients_[i]) {
                    if (std::find(byzantine_clients_.begin(), byzantine_clients_.end(),
                                 client_ids_[i]) != byzantine_clients_.end()) {
                        is_byzantine = true;
                        break;
                    }
                }
            }
            
            if (!is_byzantine) {
                values.push_back(gradient[param_idx]);
            }
        }
        
        // Compute median
        if (!values.empty()) {
            std::sort(values.begin(), values.end());
            size_t mid = values.size() / 2;
            
            if (values.size() % 2 == 0) {
                median_gradient[param_idx] = (values[mid - 1] + values[mid]) / 2.0f;
            } else {
                median_gradient[param_idx] = values[mid];
            }
        }
    }
    
    return median_gradient;
}

double GradientAggregator::computeGradientNorm(const std::vector<float>& gradient) {
    double sum_squares = 0.0;
    
    for (float val : gradient) {
        sum_squares += val * val;
    }
    
    return std::sqrt(sum_squares);
}

std::vector<int> GradientAggregator::getByzantineClients() const {
    return byzantine_clients_;
}

GradientAggregator::AggregationStats GradientAggregator::getStats() const {
    return stats_;
}

void GradientAggregator::reset() {
    gradients_.clear();
    client_ids_.clear();
    byzantine_clients_.clear();
    stats_ = AggregationStats();
}

} // namespace federated
