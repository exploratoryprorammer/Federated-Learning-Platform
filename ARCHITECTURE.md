# Federated Learning Platform - Architecture

## System Overview

The platform implements a federated learning system with three main components:

1. **Coordinator (Rust)**: Manages client registration, training rounds, and model distribution
2. **Aggregator (C++)**: Performs Byzantine-robust gradient aggregation
3. **Clients (Python)**: Train local models with differential privacy

## Component Details

### 1. Coordinator Service (Rust + gRPC)

**Location**: `coordinator/`

**Responsibilities**:

- Client registration and session management
- Training round coordination
- Global model distribution
- Gradient collection
- Heartbeat monitoring
- Prometheus metrics export

**Key Files**:

- `main.rs`: gRPC server setup
- `coordinator.rs`: Core coordination logic, round management
- `client_manager.rs`: Client tracking and lifecycle
- `metric.rs`: Prometheus metrics definitions

**gRPC Endpoints**:

```protobuf
service FederatedCoordinator {
  rpc RegisterClient(ClientRegistration) returns (ClientRegistrationResponse);
  rpc GetGlobalModel(ModelRequest) returns (ModelResponse);
  rpc SubmitGradients(GradientUpdate) returns (GradientAck);
  rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
  rpc GetTrainingConfig(ConfigRequest) returns (TrainingConfig);
}
```

**State Management**:

```rust
pub struct CoordinatorState {
    current_round: i32,
    global_model: Vec<u8>,
    client_manager: ClientManager,
    gradients_buffer: HashMap<i32, Vec<GradientUpdate>>,
    config: TrainingConfig,
}
```

**Workflow**:

1. Client registers → Assign session token and index
2. Client requests model → Serve current global model
3. Client submits gradients → Buffer until threshold reached
4. Threshold met → Trigger aggregation
5. Update model → Advance to next round

### 2. Gradient Aggregator (C++ + gRPC)

**Location**: `aggregator/`

**Responsibilities**:

- Byzantine fault-tolerant aggregation
- Gradient compression/decompression
- Outlier detection using statistical methods
- Performance-critical computation

**Key Files**:

- `aggregator.cpp`: Median-based aggregation
- `byzantine_detector.cpp`: Outlier detection (MAD-based)
- `compression.cpp`: Quantization and sparsification

**Algorithm**:

```cpp
// Coordinate-wise median aggregation
for each parameter p:
    values = [gradient[p] for all non-Byzantine clients]
    aggregated[p] = median(values)
```

**Byzantine Detection**:

```
1. Compute gradient norms for all clients
2. Calculate median norm and MAD (Median Absolute Deviation)
3. Compute modified Z-score: |norm - median| / MAD
4. Flag clients with Z-score > threshold (default: 2.5)
```

**Compression**:

- **Quantization**: 32-bit float → 8-bit integer
- **Top-K Sparsification**: Keep only K largest gradients
- **Compression Ratio**: ~75% typical

### 3. Python Clients (PyTorch)

**Location**: `python_client/`

**Responsibilities**:

- Local model training on MNIST
- Differential privacy implementation
- Gradient extraction and submission
- Communication with coordinator

**Model Architecture**:

```python
SimpleCNN:
  Conv2d(1, 32, 3x3) → ReLU → MaxPool
  Conv2d(32, 64, 3x3) → ReLU → MaxPool
  Flatten → FC(3136, 128) → ReLU → Dropout
  FC(128, 10) → Softmax
```

**Differential Privacy**:

```python
1. Clip gradients: ||g|| ≤ C (max_grad_norm)
2. Add Gaussian noise: g' = g + N(0, σ²C²)
3. Noise multiplier: σ = sqrt(2*ln(1.25/δ)) / ε
```

**Training Loop**:

```python
for round in range(total_rounds):
    1. Download global model
    2. Train locally for E epochs
    3. Extract gradients
    4. Apply DP noise
    5. Submit to coordinator
    6. Send heartbeat
```

## Communication Flow

```
┌─────────┐                    ┌─────────────┐                    ┌────────────┐
│ Client  │                    │ Coordinator │                    │ Aggregator │
└────┬────┘                    └──────┬──────┘                    └─────┬──────┘
     │                                │                                  │
     │  1. RegisterClient             │                                  │
     ├───────────────────────────────>│                                  │
     │  <session_token, client_idx>   │                                  │
     │<───────────────────────────────┤                                  │
     │                                │                                  │
     │  2. GetTrainingConfig          │                                  │
     ├───────────────────────────────>│                                  │
     │  <config>                      │                                  │
     │<───────────────────────────────┤                                  │
     │                                │                                  │
     │  3. GetGlobalModel(round=0)    │                                  │
     ├───────────────────────────────>│                                  │
     │  <model_weights>               │                                  │
     │<───────────────────────────────┤                                  │
     │                                │                                  │
     │  [Local Training]              │                                  │
     │                                │                                  │
     │  4. SubmitGradients            │                                  │
     ├───────────────────────────────>│                                  │
     │                                │  [Buffer gradients]              │
     │                                │                                  │
     │  5. SubmitGradients (client 2) │                                  │
     │                                │<─────────────────────────────────│
     │                                │                                  │
     │  [Threshold reached]           │                                  │
     │                                │  6. AggregateGradients           │
     │                                ├─────────────────────────────────>│
     │                                │                                  │
     │                                │  [Byzantine detection]           │
     │                                │  [Median aggregation]            │
     │                                │                                  │
     │                                │  <aggregated_weights>            │
     │                                │<─────────────────────────────────┤
     │                                │                                  │
     │                                │  [Update global model]           │
     │                                │  [Advance round]                 │
     │                                │                                  │
     │  7. Heartbeat                  │                                  │
     ├───────────────────────────────>│                                  │
     │  <continue=true, next_round=1> │                                  │
     │<───────────────────────────────┤                                  │
     │                                │                                  │
```

## Data Structures

### Protocol Buffers

**ClientRegistration**:

```protobuf
message ClientRegistration {
  string client_id = 1;
  string client_version = 2;
  ClientCapabilities capabilities = 3;
  int64 timestamp = 4;
}
```

**GradientUpdate**:

```protobuf
message GradientUpdate {
  string client_id = 1;
  string session_token = 2;
  int32 round_number = 3;
  bytes gradients = 4;  // Serialized numpy array
  GradientMetadata metadata = 5;
  int64 timestamp = 6;
}
```

**ModelResponse**:

```protobuf
message ModelResponse {
  int32 round_number = 1;
  bytes model_weights = 2;
  int64 model_version = 3;
  ModelMetadata metadata = 4;
}
```

## Security & Privacy

### Differential Privacy

**Privacy Budget**: (ε=2.0, δ=1e-5)

- ε (epsilon): Privacy loss parameter (lower = more private)
- δ (delta): Failure probability

**Mechanism**:

1. **Gradient Clipping**: Bound sensitivity

   ```
   g_clipped = g / max(1, ||g|| / C)
   ```

2. **Noise Addition**: Gaussian mechanism
   ```
   g_private = g_clipped + N(0, σ²C²)
   ```

**Privacy Accounting**:

- Per-round privacy: (ε_r, δ_r)
- Total privacy: Composition over T rounds
- Advanced composition: ε_total ≈ ε_r _ sqrt(2T _ ln(1/δ))

### Byzantine Fault Tolerance

**Threat Model**:

- Up to f < n/2 malicious clients
- Adversary can send arbitrary gradients
- Goal: Prevent model poisoning

**Defense**:

- **Median Aggregation**: Robust to outliers
- **MAD-based Detection**: Statistical outlier identification
- **Threshold**: Z-score > 2.5 (configurable)

**Limitations**:

- Assumes majority of clients are honest
- May reduce convergence speed
- Trade-off between robustness and accuracy

## Performance Optimization

### Rust Coordinator

**Async I/O**: Tokio runtime

```rust
#[tokio::main]
async fn main() {
    Server::builder()
        .add_service(FederatedCoordinatorServer::new(service))
        .serve(addr)
        .await
}
```

**Concurrent State Access**: RwLock

```rust
Arc<RwLock<CoordinatorState>>
```

### C++ Aggregator

**Parallelization**: OpenMP

```cpp
#pragma omp parallel for
for (int i = 0; i < num_parameters; ++i) {
    median_gradient[i] = compute_median(values[i]);
}
```

**SIMD**: Eigen library for vectorized operations

**Memory Layout**: Contiguous arrays for cache efficiency

### Python Client

**GPU Acceleration**: CUDA if available

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Data Loading**: Multi-threaded DataLoader

```python
DataLoader(dataset, batch_size=32, num_workers=4)
```

## Monitoring & Observability

### Prometheus Metrics

**Coordinator**:

- `federated_active_clients`: Gauge
- `federated_current_round`: Gauge
- `federated_model_accuracy`: Gauge
- `federated_gradients_received_total`: Counter
- `federated_aggregation_seconds`: Histogram

**Queries**:

```promql
# Client participation rate
rate(federated_client_participation_total[5m])

# Aggregation latency (p95)
histogram_quantile(0.95, federated_aggregation_seconds_bucket)

# Byzantine detection rate
rate(federated_byzantine_clients_detected_total[5m])
```

### Logging

**Rust**: tracing crate

```rust
tracing::info!("Round {} complete: loss={:.4}", round, loss);
```

**C++**: std::cout with timestamps

```cpp
std::cout << "Aggregating " << n << " gradients..." << std::endl;
```

**Python**: logging module

```python
logger.info(f"Training complete: accuracy={acc:.2f}%")
```

## Scalability

### Current Limits

- **Clients**: 100 (configurable)
- **Parameters**: ~10K (MNIST model)
- **Round Time**: ~15-20s with 5 clients

### Scaling Strategies

**Horizontal**:

- Multiple aggregator instances
- Load balancing across coordinators
- Sharded client pools

**Vertical**:

- Larger models (ResNet, Transformers)
- More clients per round
- Faster hardware (GPU aggregation)

**Optimizations**:

- Asynchronous aggregation
- Hierarchical aggregation (edge → cloud)
- Gradient compression (quantization, sparsification)

## Deployment

### Development

```bash
./scripts/setup.sh
./scripts/start_all.sh
```

### Production (Docker)

```bash
docker-compose up -d
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coordinator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: coordinator
  template:
    spec:
      containers:
        - name: coordinator
          image: federated/coordinator:latest
          ports:
            - containerPort: 50051
            - containerPort: 9090
```

## Future Enhancements

1. **Secure Aggregation**: Cryptographic protocols for privacy
2. **Adaptive Privacy**: Dynamic ε based on data sensitivity
3. **Client Selection**: Intelligent sampling strategies
4. **Model Compression**: Pruning, quantization for edge devices
5. **Cross-Silo FL**: Support for organizational boundaries
6. **Personalization**: Local model fine-tuning
7. **Asynchronous FL**: Remove synchronization barriers

## References

- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
- Abadi et al. "Deep Learning with Differential Privacy" (2016)
- Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (2017)
