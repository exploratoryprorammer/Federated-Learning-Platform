#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace federated {

class GradientAggregator {
public:
    GradientAggregator(int num_parameters, int expected_clients);
    
    // Add gradient from a client
    void addGradient(const std::vector<float>& gradient, int client_id);
    
    // Perform aggregation with Byzantine fault tolerance
    std::vector<float> aggregate();
    
    // Get detected Byzantine clients
    std::vector<int> getByzantineClients() const;
    
    // Get aggregation statistics
    struct AggregationStats {
        double aggregation_time_ms;
        double compression_ratio;
        int outliers_detected;
        double median_gradient_norm;
        std::vector<double> client_gradient_norms;
    };
    
    AggregationStats getStats() const;
    
    // Reset for next round
    void reset();

private:
    int num_parameters_;
    int expected_clients_;
    std::vector<std::vector<float>> gradients_;
    std::vector<int> client_ids_;
    std::vector<int> byzantine_clients_;
    AggregationStats stats_;
    
    // Helper methods
    double computeGradientNorm(const std::vector<float>& gradient);
    std::vector<float> computeMedianGradient();
};

}
