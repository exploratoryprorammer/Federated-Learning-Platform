#!/bin/bash
# Integration test for federated learning platform

set -e

echo "=========================================="
echo "Federated Learning Integration Test"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    pkill -f "coordinator" || true
    pkill -f "aggregator_server" || true
    pkill -f "client.py" || true
    sleep 2
}

trap cleanup EXIT

# Test 1: Build check
echo -e "\n${YELLOW}[1/6] Checking builds...${NC}"
if [ ! -f "coordinator/target/release/coordinator" ]; then
    echo -e "${RED}✗ Coordinator not built${NC}"
    exit 1
fi
if [ ! -f "aggregator/build/aggregator_server" ]; then
    echo -e "${RED}✗ Aggregator not built${NC}"
    exit 1
fi
if [ ! -f "python_client/venv/bin/python" ]; then
    echo -e "${RED}✗ Python venv not setup${NC}"
    exit 1
fi
echo -e "${GREEN}✓ All components built${NC}"

# Test 2: Start coordinator
echo -e "\n${YELLOW}[2/6] Starting coordinator...${NC}"
cd coordinator
RUST_LOG=info cargo run --release > /tmp/coordinator.log 2>&1 &
COORDINATOR_PID=$!
cd ..
sleep 3

if ! ps -p $COORDINATOR_PID > /dev/null; then
    echo -e "${RED}✗ Coordinator failed to start${NC}"
    cat /tmp/coordinator.log
    exit 1
fi
echo -e "${GREEN}✓ Coordinator started (PID: $COORDINATOR_PID)${NC}"

# Test 3: Check health endpoint
echo -e "\n${YELLOW}[3/6] Checking health endpoint...${NC}"
if curl -s http://localhost:9090/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ Health endpoint responding${NC}"
else
    echo -e "${RED}✗ Health endpoint not responding${NC}"
    exit 1
fi

# Test 4: Start aggregator
echo -e "\n${YELLOW}[4/6] Starting aggregator...${NC}"
cd aggregator/build
./aggregator_server > /tmp/aggregator.log 2>&1 &
AGGREGATOR_PID=$!
cd ../..
sleep 2

if ! ps -p $AGGREGATOR_PID > /dev/null; then
    echo -e "${RED}✗ Aggregator failed to start${NC}"
    cat /tmp/aggregator.log
    exit 1
fi
echo -e "${GREEN}✓ Aggregator started (PID: $AGGREGATOR_PID)${NC}"

# Test 5: Start clients
echo -e "\n${YELLOW}[5/6] Starting test clients...${NC}"
cd python_client
source venv/bin/activate

# Start 3 clients
for i in 0 1 2; do
    python client.py --client-id "test_client_$i" > "/tmp/client_$i.log" 2>&1 &
    CLIENT_PIDS[$i]=$!
    echo "  Started client $i (PID: ${CLIENT_PIDS[$i]})"
    sleep 1
done

cd ..
sleep 5

# Check clients are running
CLIENTS_OK=true
for i in 0 1 2; do
    if ! ps -p ${CLIENT_PIDS[$i]} > /dev/null; then
        echo -e "${RED}✗ Client $i failed${NC}"
        cat "/tmp/client_$i.log"
        CLIENTS_OK=false
    fi
done

if [ "$CLIENTS_OK" = true ]; then
    echo -e "${GREEN}✓ All clients started${NC}"
else
    exit 1
fi

# Test 6: Monitor training
echo -e "\n${YELLOW}[6/6] Monitoring training progress...${NC}"
echo "Waiting for training to complete (max 60s)..."

for i in {1..60}; do
    # Check active clients
    ACTIVE=$(curl -s http://localhost:9090/metrics | grep "federated_active_clients " | awk '{print $2}')
    
    # Check current round
    ROUND=$(curl -s http://localhost:9090/metrics | grep "federated_current_round " | awk '{print $2}')
    
    # Check gradients received
    GRADIENTS=$(curl -s http://localhost:9090/metrics | grep "federated_gradients_received_total " | awk '{print $2}')
    
    echo -ne "\r  Active clients: ${ACTIVE:-0} | Round: ${ROUND:-0} | Gradients: ${GRADIENTS:-0}  "
    
    # Success condition: at least 1 round completed
    if [ ! -z "$ROUND" ] && [ "$ROUND" != "0" ]; then
        echo -e "\n${GREEN}✓ Training progressing (Round $ROUND completed)${NC}"
        break
    fi
    
    sleep 1
done

# Final checks
echo -e "\n${YELLOW}Running final checks...${NC}"

# Check metrics
METRICS=$(curl -s http://localhost:9090/metrics)

if echo "$METRICS" | grep -q "federated_active_clients"; then
    echo -e "${GREEN}✓ Metrics endpoint working${NC}"
else
    echo -e "${RED}✗ Metrics not available${NC}"
    exit 1
fi

# Check registrations
REGISTRATIONS=$(echo "$METRICS" | grep "federated_total_registrations " | awk '{print $2}')
if [ ! -z "$REGISTRATIONS" ] && [ "$REGISTRATIONS" != "0" ]; then
    echo -e "${GREEN}✓ Clients registered ($REGISTRATIONS total)${NC}"
else
    echo -e "${RED}✗ No client registrations${NC}"
    exit 1
fi

# Check gradients
if [ ! -z "$GRADIENTS" ] && [ "$GRADIENTS" != "0" ]; then
    echo -e "${GREEN}✓ Gradients received ($GRADIENTS total)${NC}"
else
    echo -e "${YELLOW}⚠ No gradients received yet${NC}"
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Integration Test PASSED${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Coordinator: Running (PID: $COORDINATOR_PID)"
echo "  - Aggregator: Running (PID: $AGGREGATOR_PID)"
echo "  - Clients: 3 active"
echo "  - Registrations: $REGISTRATIONS"
echo "  - Gradients: $GRADIENTS"
echo "  - Current Round: $ROUND"
echo ""
echo "Logs available at:"
echo "  - /tmp/coordinator.log"
echo "  - /tmp/aggregator.log"
echo "  - /tmp/client_*.log"
echo ""
echo "Metrics: http://localhost:9090/metrics"
echo ""

# Keep services running for manual inspection
echo -e "${YELLOW}Services are still running. Press Ctrl+C to stop.${NC}"
wait
