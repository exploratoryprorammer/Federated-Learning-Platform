# Quick Start Guide

Get your federated learning platform running in 5 minutes!

## Prerequisites

- macOS or Linux
- Rust 1.70+ ([Install](https://rustup.rs/))
- C++17 compiler (Xcode on macOS, g++ on Linux)
- CMake 3.15+
- Python 3.8+
- 4GB RAM minimum

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone and setup
git clone <your-repo>
cd federated-learning-platform

# Run setup script (builds everything)
./scripts/setup.sh
```

This will:

1. Check all prerequisites
2. Generate Protocol Buffer code
3. Build Rust coordinator
4. Build C++ aggregator
5. Setup Python virtual environment

### Option 2: Manual Setup

```bash
# 1. Generate protobuf code
cd python_client
python3 -m venv venv
source venv/bin/activate
pip install grpcio-tools
cd ../proto
python3 -m grpc_tools.protoc -I. --python_out=../python_client --grpc_python_out=../python_client coordinator.proto

# 2. Build coordinator
cd ../coordinator
cargo build --release

# 3. Build aggregator
cd ../aggregator
mkdir build && cd build
cmake ..
make

# 4. Install Python dependencies
cd ../../python_client
source venv/bin/activate
pip install -r requirements.txt
```

## Running the System

### Option A: All-in-One (tmux)

```bash
./scripts/start_all.sh
```

This starts everything in a tmux session. Press `Ctrl+B` then `D` to detach.

### Option B: Manual (Separate Terminals)

**Terminal 1 - Coordinator:**

```bash
cd coordinator
cargo run --release
```

**Terminal 2 - Aggregator:**

```bash
cd aggregator/build
./aggregator_server
```

**Terminal 3-7 - Clients (5 clients):**

```bash
cd python_client
source venv/bin/activate

# Run in separate terminals
python client.py --client-id client_0
python client.py --client-id client_1
python client.py --client-id client_2
python client.py --client-id client_3
python client.py --client-id client_4
```

## Verify It's Working

### 1. Check Services

```bash
# Coordinator health
curl http://localhost:9090/health

# Metrics
curl http://localhost:9090/metrics | grep federated_active_clients
```

### 2. Watch Training Progress

```bash
# Monitor metrics in real-time
watch -n 2 'curl -s http://localhost:9090/metrics | grep -E "(active_clients|current_round|model_accuracy)"'
```

### 3. Expected Output

**Coordinator logs:**

```
INFO Coordinator listening on 0.0.0.0:50051
INFO Metrics server listening on: http://0.0.0.0:9090/metrics
INFO Registering client: client_0
INFO Client client_0 registered with index 0
INFO Received gradients from client_0 for round 0
INFO Threshold reached. Triggering aggregation for round 0
INFO Round 0 complete: loss=0.5234, accuracy=89.23%, time=45.23ms
```

**Client logs:**

```
INFO Client client_0 initialized on device: cpu
INFO Registration successful: Successfully registered as client 0
INFO Training config received:
INFO   - Total rounds: 10
INFO   - Local epochs: 5
INFO Loaded 12000 training samples
INFO Starting local training...
INFO   Epoch 1/5: Loss = 0.6234
INFO Local training complete: Loss = 0.5234, Accuracy = 89.23%
INFO Gradients accepted: Gradient received (3/3)
```

## Training Configuration

Edit `coordinator/src/coordinator.rs`:

```rust
fn default_config() -> TrainingConfig {
    TrainingConfig {
        total_rounds: 10,           // ← Change number of rounds
        local_epochs: 5,            // ← Epochs per client
        batch_size: 32,
        learning_rate: 0.01,
        enable_differential_privacy: true,
        aggregation_threshold: 3,   // ← Min clients per round
        // ...
    }
}
```

## Monitoring

### Prometheus Metrics

Visit: http://localhost:9090/metrics

Key metrics:

- `federated_active_clients` - Connected clients
- `federated_current_round` - Training progress
- `federated_model_accuracy` - Model performance
- `federated_gradients_received_total` - Total submissions

### Query Examples

```bash
# Active clients
curl -s http://localhost:9090/metrics | grep "federated_active_clients "

# Current round
curl -s http://localhost:9090/metrics | grep "federated_current_round "

# Model accuracy
curl -s http://localhost:9090/metrics | grep "federated_model_accuracy "
```

## Troubleshooting

### "Address already in use"

```bash
# Find and kill process on port 50051
lsof -ti:50051 | xargs kill -9

# Or use different port
cd coordinator/src
# Edit main.rs: change "0.0.0.0:50051" to "0.0.0.0:50052"
```

### "Connection refused"

```bash
# Check coordinator is running
curl http://localhost:9090/health

# Check logs
cd coordinator
RUST_LOG=debug cargo run
```

### "No module named coordinator_pb2"

```bash
# Regenerate protobuf code
cd proto
python3 -m grpc_tools.protoc -I. --python_out=../python_client --grpc_python_out=../python_client coordinator.proto
```

### Clients not connecting

```bash
# Verify coordinator is listening
netstat -an | grep 50051

# Check firewall
# macOS: System Preferences → Security & Privacy → Firewall
# Linux: sudo ufw status
```

### Low accuracy

- Increase `local_epochs` (more training per round)
- Decrease `epsilon` in DP config (less noise, but less privacy)
- Increase `aggregation_threshold` (more clients per round)
- Train for more rounds

## Next Steps

### 1. Customize the Model

Edit `python_client/client.py`:

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Modify architecture here
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)  # More filters
        # ...
```

### 2. Use Different Dataset

```python
# In client.py, replace MNIST with:
train_dataset = datasets.CIFAR10(
    data_dir, train=True, download=True, transform=transform
)
```

### 3. Adjust Privacy Budget

```python
# In client.py
self.dp = DifferentialPrivacy(
    epsilon=1.0,        # Lower = more privacy
    delta=1e-5,
    max_grad_norm=0.5   # Stricter clipping
)
```

### 4. Add More Clients

```bash
# Just run with different IDs
python client.py --client-id client_5
python client.py --client-id client_6
# ...
```

### 5. Deploy with Docker

```bash
docker-compose up -d
```

## Performance Tips

### Speed up training

1. **Use GPU**: Clients automatically use CUDA if available
2. **Reduce local_epochs**: Faster rounds, but may need more rounds
3. **Increase batch_size**: Better GPU utilization
4. **Enable compression**: Reduce network overhead

### Reduce memory usage

1. **Smaller batch_size**: Less memory per client
2. **Fewer clients**: Run 2-3 instead of 5
3. **Simpler model**: Reduce CNN layers

## Common Use Cases

### 1. Quick Test (1 round, 2 clients)

```rust
// coordinator/src/coordinator.rs
total_rounds: 1,
aggregation_threshold: 2,
```

```bash
python client.py --client-id client_0 &
python client.py --client-id client_1 &
```

### 2. High Privacy (ε=0.5)

```python
# python_client/client.py
self.dp = DifferentialPrivacy(epsilon=0.5, delta=1e-5, max_grad_norm=0.5)
```

### 3. No Privacy (Faster)

```bash
python client.py --client-id client_0 --no-dp
```

### 4. Production Deployment

```bash
# Use Docker Compose
docker-compose up -d

# Or Kubernetes
kubectl apply -f k8s/
```

## Getting Help

- Check logs: `RUST_LOG=debug cargo run` (coordinator)
- Enable verbose: `./aggregator_server --verbose` (aggregator)
- Python debug: `logging.basicConfig(level=logging.DEBUG)` (client)

## What's Next?

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [README.md](README.md) for detailed documentation
- Explore metrics at http://localhost:9090/metrics
- Set up Grafana dashboards for visualization

## Success Checklist

- [ ] All services start without errors
- [ ] Clients register successfully
- [ ] Training rounds complete
- [ ] Metrics endpoint responds
- [ ] Accuracy improves over rounds
- [ ] No Byzantine clients detected (in normal operation)

Congratulations! Your federated learning platform is running! 🎉
