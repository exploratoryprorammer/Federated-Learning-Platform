#include <iostream>
#include <grpcpp/grpcpp.h>
#include <memory>
#include "aggregator.h"
#include "federated.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

class GradientAggregatorService final : public federated::GradientAggregator::Service {
public:
    GradientAggregatorService(int num_parameters, int expected_clients)
        : aggregator_(num_parameters, expected_clients) {}

    Status AggregateGradients(
        ServerContext* context,
        grpc::ServerReader<federated::GradientBatch>* reader,
        federated::AggregatedGradients* response) override {
        
        aggregator_.reset();
        
        federated::GradientBatch batch;
        while (reader->Read(&batch)) {
            for (const auto& update : batch.updates()) {
                // Deserialize gradients (simplified)
                std::vector<float> gradient;
                // In production: properly deserialize from update.gradients()
                
                aggregator_.addGradient(gradient, 
                    std::stoi(update.client_id()));
            }
        }
        
        // Perform aggregation
        std::vector<float> aggregated = aggregator_.aggregate();
        
        // Serialize result
        response->set_aggregated_weights(
            reinterpret_cast<const char*>(aggregated.data()),
            aggregated.size() * sizeof(float)
        );
        
        response->set_num_clients_aggregated(
            aggregator_.getStats().client_gradient_norms.size() - 
            aggregator_.getByzantineClients().size()
        );
        
        for (int byzantine_id : aggregator_.getByzantineClients()) {
            response->add_byzantine_clients(byzantine_id);
        }
        
        // Set metadata
        auto* metadata = response->mutable_metadata();
        metadata->set_aggregation_time_ms(aggregator_.getStats().aggregation_time_ms);
        metadata->set_outliers_detected(aggregator_.getStats().outliers_detected);
        metadata->set_median_gradient_norm(aggregator_.getStats().median_gradient_norm);
        
        return Status::OK;
    }

    Status GetAggregationStats(
        ServerContext* context,
        const federated::StatsRequest* request,
        federated::AggregationStats* response) override {
        
        auto stats = aggregator_.getStats();
        
        response->set_total_gradients_processed(
            stats.client_gradient_norms.size()
        );
        response->set_average_aggregation_time_ms(stats.aggregation_time_ms);
        response->set_total_byzantine_detected(stats.outliers_detected);
        
        return Status::OK;
    }

private:
    federated::GradientAggregator aggregator_;
};

void RunServer() {
    std::string server_address("0.0.0.0:50052");
    
    // Default: 784 * 10 = 7840 parameters for simple MNIST model
    GradientAggregatorService service(7840, 5);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Gradient Aggregator listening on " << server_address << std::endl;

    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}