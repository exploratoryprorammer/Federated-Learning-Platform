#!/bin/bash
# Setup script for federated learning platform

set -e

echo "=========================================="
echo "Federated Learning Platform Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Error: Rust is not installed${NC}"
    echo "Install from: https://rustup.rs/"
    exit 1
fi
echo -e "${GREEN}✓ Rust found: $(rustc --version)${NC}"

# Check C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo -e "${RED}Error: C++ compiler not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ C++ compiler found${NC}"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake is not installed${NC}"
    echo "Install with: brew install cmake (macOS) or apt-get install cmake (Linux)"
    exit 1
fi
echo -e "${GREEN}✓ CMake found: $(cmake --version | head -n1)${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Check protoc
if ! command -v protoc &> /dev/null; then
    echo -e "${YELLOW}Warning: protoc not found. Installing via pip...${NC}"
fi

echo ""
echo -e "${YELLOW}Building components...${NC}"

# 1. Generate Protocol Buffers
echo -e "\n${YELLOW}[1/4] Generating Protocol Buffers...${NC}"
cd python_client
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install grpcio-tools
cd ../proto
python3 -m grpc_tools.protoc -I. \
    --python_out=../python_client \
    --grpc_python_out=../python_client \
    coordinator.proto
echo -e "${GREEN}✓ Protocol Buffers generated${NC}"

# 2. Build Rust coordinator
echo -e "\n${YELLOW}[2/4] Building Rust coordinator...${NC}"
cd ../coordinator
cargo build --release
echo -e "${GREEN}✓ Coordinator built${NC}"

# 3. Build C++ aggregator
echo -e "\n${YELLOW}[3/4] Building C++ aggregator...${NC}"
cd ../aggregator
mkdir -p build
cd build
cmake ..
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
echo -e "${GREEN}✓ Aggregator built${NC}"

# 4. Setup Python client
echo -e "\n${YELLOW}[4/4] Setting up Python client...${NC}"
cd ../../python_client
source venv/bin/activate
pip install -r requirements.txt
echo -e "${GREEN}✓ Python client setup complete${NC}"

# Make scripts executable
cd ..
chmod +x scripts/*.sh

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Start all services: ./scripts/start_all.sh"
echo "  2. Or manually:"
echo "     - Terminal 1: cd coordinator && cargo run --release"
echo "     - Terminal 2: cd aggregator/build && ./aggregator_server"
echo "     - Terminal 3+: cd python_client && source venv/bin/activate && python client.py --client-id client_X"
echo ""
echo "Monitor metrics at: http://localhost:9090/metrics"
echo ""
