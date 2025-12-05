.PHONY: all build clean proto coordinator aggregator client test run help

# Default target
all: build

help:
	@echo "Federated Learning Platform - Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make build        - Build all components"
	@echo "  make proto        - Generate protobuf code"
	@echo "  make coordinator  - Build Rust coordinator"
	@echo "  make aggregator   - Build C++ aggregator"
	@echo "  make client       - Setup Python client"
	@echo "  make run          - Run all services"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean build artifacts"

# Build everything
build: proto coordinator aggregator client

# Generate Protocol Buffer code
proto:
	@echo "Generating Protocol Buffer code..."
	@cd proto && python3 -m grpc_tools.protoc -I. \
		--python_out=../python_client \
		--grpc_python_out=../python_client \
		coordinator.proto
	@echo "✓ Protobuf code generated"

# Build Rust coordinator
coordinator:
	@echo "Building Rust coordinator..."
	@cd coordinator && cargo build --release
	@echo "✓ Coordinator built"

# Build C++ aggregator
aggregator:
	@echo "Building C++ aggregator..."
	@mkdir -p aggregator/build
	@cd aggregator/build && cmake .. && make
	@echo "✓ Aggregator built"

# Setup Python client
client:
	@echo "Setting up Python client..."
	@cd python_client && \
		python3 -m venv venv && \
		. venv/bin/activate && \
		pip install -r requirements.txt
	@echo "✓ Python client setup complete"

# Run tests
test:
	@echo "Running tests..."
	@cd coordinator && cargo test
	@echo "✓ Tests passed"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@cd coordinator && cargo clean
	@rm -rf aggregator/build
	@rm -rf python_client/venv
	@rm -f python_client/*_pb2.py python_client/*_pb2_grpc.py
	@echo "✓ Clean complete"

# Run all services (requires tmux)
run:
	@echo "Starting all services..."
	@echo "This requires tmux. Press Ctrl+B then D to detach."
	@tmux new-session -d -s federated 'cd coordinator && cargo run --release'
	@tmux split-window -h 'cd aggregator/build && ./aggregator_server'
	@tmux split-window -v 'cd python_client && . venv/bin/activate && python client.py --client-id client_0'
	@tmux attach-session -t federated
