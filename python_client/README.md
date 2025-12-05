# Federated Learning Python Client

PyTorch-based federated learning client with differential privacy support.

## Setup

```bash
cd python_client
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Generate Protocol Buffers

```bash
cd ../proto
python -m grpc_tools.protoc -I. --python_out=../python_client --grpc_python_out=../python_client coordinator.proto
```

## Run Client

```bash
# Start multiple clients (in separate terminals)
python client.py --client-id client_0
python client.py --client-id client_1
python client.py --client-id client_2
python client.py --client-id client_3
python client.py --client-id client_4
```

## Features

- **PyTorch Training**: MNIST classification with SimpleCNN
- **Differential Privacy**: Gradient clipping and noise injection
- **gRPC Communication**: Efficient client-server communication
- **Non-IID Data**: Simulates realistic federated scenarios
