#!/bin/bash
# Start all federated learning services

set -e

echo "=========================================="
echo "Federated Learning Platform Startup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux is not installed${NC}"
    echo "Install with: brew install tmux (macOS) or apt-get install tmux (Linux)"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t federated 2>/dev/null || true

echo -e "${GREEN}Starting services in tmux session 'federated'...${NC}"
echo ""
echo "Commands:"
echo "  - Ctrl+B then D: Detach from session"
echo "  - tmux attach -t federated: Reattach"
echo "  - tmux kill-session -t federated: Stop all services"
echo ""

# Create new session with coordinator
tmux new-session -d -s federated -n coordinator "cd coordinator && cargo run --release; read"

# Split window for aggregator
tmux split-window -h -t federated "cd aggregator/build && ./aggregator_server; read"

# Split for metrics monitoring
tmux split-window -v -t federated "sleep 3 && watch -n 2 'curl -s http://localhost:9090/metrics | grep federated_active_clients'; read"

# Create new window for clients
tmux new-window -t federated -n clients

# Split into 5 panes for clients
tmux split-window -h -t federated:clients
tmux split-window -v -t federated:clients
tmux select-pane -t federated:clients.0
tmux split-window -v -t federated:clients

# Start clients
tmux send-keys -t federated:clients.0 "cd python_client && source venv/bin/activate && sleep 5 && python client.py --client-id client_0" C-m
tmux send-keys -t federated:clients.1 "cd python_client && source venv/bin/activate && sleep 5 && python client.py --client-id client_1" C-m
tmux send-keys -t federated:clients.2 "cd python_client && source venv/bin/activate && sleep 5 && python client.py --client-id client_2" C-m
tmux send-keys -t federated:clients.3 "cd python_client && source venv/bin/activate && sleep 5 && python client.py --client-id client_3" C-m

echo -e "${GREEN}✓ All services started!${NC}"
echo ""
echo "Attaching to tmux session..."
sleep 2

# Attach to session
tmux attach-session -t federated
