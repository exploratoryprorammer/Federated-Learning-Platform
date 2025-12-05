#!/usr/bin/env python3
"""
Federated Learning Client with PyTorch
Implements privacy-preserving distributed training with differential privacy
"""

import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time
import argparse
import logging
from typing import Tuple, Optional
import sys
import os

# Add proto directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'proto'))

# Import generated protobuf code
import coordinator_pb2
import coordinator_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DifferentialPrivacy:
    """Differential Privacy noise injection for gradients"""
    
    def __init__(self, epsilon: float = 2.0, delta: float = 1e-5, 
                 max_grad_norm: float = 1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = self._compute_noise_multiplier()
    
    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier from privacy budget"""
        # Simplified calculation (in production, use proper DP accounting)
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients to max norm"""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_grad_norm
        )
        return total_norm.item()
    
    def add_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to gradients"""
        noise = torch.normal(
            mean=0.0,
            std=self.noise_multiplier * self.max_grad_norm,
            size=gradients.shape
        )
        return gradients + noise


class FederatedClient:
    """Federated Learning Client"""
    
    def __init__(self, client_id: str, server_address: str = 'localhost:50051',
                 enable_dp: bool = True):
        self.client_id = client_id
        self.server_address = server_address
        self.enable_dp = enable_dp
        
        # gRPC channel and stub
        self.channel = grpc.insecure_channel(server_address)
        self.stub = coordinator_pb2_grpc.FederatedCoordinatorStub(self.channel)
        
        # Session info
        self.session_token: Optional[str] = None
        self.client_index: Optional[int] = None
        
        # Model and training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # Differential privacy
        if enable_dp:
            self.dp = DifferentialPrivacy(epsilon=2.0, delta=1e-5, max_grad_norm=1.0)
        else:
            self.dp = None
        
        # Training config
        self.config: Optional[coordinator_pb2.TrainingConfig] = None
        
        logger.info(f"Client {client_id} initialized on device: {self.device}")
    
    def register(self) -> bool:
        """Register with coordinator"""
        try:
            # Get system info
            num_samples = 1000  # Will be set after loading data
            
            capabilities = coordinator_pb2.ClientCapabilities(
                num_samples=num_samples,
                supports_compression=True,
                supports_differential_privacy=self.enable_dp,
                hardware_info=f"PyTorch {torch.__version__}, Device: {self.device}"
            )
            
            request = coordinator_pb2.ClientRegistration(
                client_id=self.client_id,
                client_version="1.0.0",
                capabilities=capabilities,
                timestamp=int(time.time())
            )
            
            response = self.stub.RegisterClient(request)
            
            if response.success:
                self.session_token = response.session_token
                self.client_index = response.assigned_client_index
                logger.info(f"Registration successful: {response.message}")
                logger.info(f"Assigned index: {self.client_index}")
                return True
            else:
                logger.error(f"Registration failed: {response.message}")
                return False
                
        except grpc.RpcError as e:
            logger.error(f"Registration RPC failed: {e}")
            return False
    
    def get_training_config(self) -> bool:
        """Get training configuration from coordinator"""
        try:
            request = coordinator_pb2.ConfigRequest(client_id=self.client_id)
            self.config = self.stub.GetTrainingConfig(request)
            
            logger.info(f"Training config received:")
            logger.info(f"  - Total rounds: {self.config.total_rounds}")
            logger.info(f"  - Local epochs: {self.config.local_epochs}")
            logger.info(f"  - Batch size: {self.config.batch_size}")
            logger.info(f"  - Learning rate: {self.config.learning_rate}")
            logger.info(f"  - Differential Privacy: {self.config.enable_differential_privacy}")
            
            return True
            
        except grpc.RpcError as e:
            logger.error(f"Failed to get config: {e}")
            return False
    
    def load_data(self, data_dir: str = './data') -> Tuple[DataLoader, DataLoader]:
        """Load and partition MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download MNIST
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=transform
        )
        
        # Partition data for this client (non-IID simulation)
        # Each client gets a subset of the data
        total_clients = 5  # Assuming 5 clients
        samples_per_client = len(train_dataset) // total_clients
        
        start_idx = self.client_index * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client_indices = list(range(start_idx, end_idx))
        client_train = Subset(train_dataset, client_indices)
        
        # Create data loaders
        batch_size = self.config.batch_size if self.config else 32
        
        train_loader = DataLoader(
            client_train, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        logger.info(f"Loaded {len(client_train)} training samples")
        
        return train_loader, test_loader
    
    def get_global_model(self, round_number: int) -> bool:
        """Download global model from coordinator"""
        try:
            request = coordinator_pb2.ModelRequest(
                client_id=self.client_id,
                session_token=self.session_token,
                current_round=round_number
            )
            
            response = self.stub.GetGlobalModel(request)
            
            # Deserialize model weights
            # In production: properly deserialize PyTorch state dict
            logger.info(f"Received global model for round {response.round_number}")
            logger.info(f"Model version: {response.model_version}")
            
            return True
            
        except grpc.RpcError as e:
            logger.error(f"Failed to get global model: {e}")
            return False
    
    def train_local_model(self, train_loader: DataLoader, 
                         epochs: int) -> Tuple[float, float]:
        """Train model locally"""
        self.model.train()
        
        learning_rate = self.config.learning_rate if self.config else 0.01
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                # Apply differential privacy
                if self.dp:
                    grad_norm = self.dp.clip_gradients(self.model)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"  Epoch {epoch + 1}/{epochs}: Loss = {avg_epoch_loss:.4f}")
            total_loss += avg_epoch_loss
        
        avg_loss = total_loss / epochs
        accuracy = 100.0 * correct / total
        
        logger.info(f"Local training complete: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
        
        return avg_loss, accuracy
    
    def extract_gradients(self) -> bytes:
        """Extract gradients from model"""
        gradients = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                grad = param.grad.data.cpu().numpy().flatten()
                
                # Apply differential privacy noise
                if self.dp:
                    grad_tensor = torch.from_numpy(grad)
                    grad_tensor = self.dp.add_noise(grad_tensor)
                    grad = grad_tensor.numpy()
                
                gradients.append(grad)
        
        # Concatenate all gradients
        all_gradients = np.concatenate(gradients)
        
        # Serialize to bytes
        return all_gradients.astype(np.float32).tobytes()
    
    def submit_gradients(self, round_number: int, loss: float, 
                        accuracy: float, num_samples: int) -> bool:
        """Submit gradients to coordinator"""
        try:
            # Extract gradients
            gradients_bytes = self.extract_gradients()
            
            # Create metadata
            metadata = coordinator_pb2.GradientMetadata(
                num_samples_trained=num_samples,
                local_loss=loss,
                local_accuracy=accuracy,
                is_compressed=False,
                compression_ratio=1.0,
                has_differential_privacy=self.enable_dp
            )
            
            if self.enable_dp and self.dp:
                metadata.dp_params.CopyFrom(coordinator_pb2.DPParameters(
                    epsilon=self.dp.epsilon,
                    delta=self.dp.delta,
                    noise_multiplier=self.dp.noise_multiplier,
                    max_grad_norm=self.dp.max_grad_norm
                ))
            
            # Create gradient update
            update = coordinator_pb2.GradientUpdate(
                client_id=self.client_id,
                session_token=self.session_token,
                round_number=round_number,
                gradients=gradients_bytes,
                metadata=metadata,
                timestamp=int(time.time())
            )
            
            # Submit
            response = self.stub.SubmitGradients(update)
            
            if response.accepted:
                logger.info(f"Gradients accepted: {response.message}")
                return True
            else:
                logger.warning(f"Gradients rejected: {response.message}")
                return False
                
        except grpc.RpcError as e:
            logger.error(f"Failed to submit gradients: {e}")
            return False
    
    def send_heartbeat(self, current_round: int, is_training: bool = False) -> bool:
        """Send heartbeat to coordinator"""
        try:
            status = coordinator_pb2.ClientStatus(
                is_training=is_training,
                cpu_usage=0.0,  # Could use psutil to get real values
                memory_usage_mb=0.0,
                current_round=current_round
            )
            
            request = coordinator_pb2.HeartbeatRequest(
                client_id=self.client_id,
                session_token=self.session_token,
                status=status
            )
            
            response = self.stub.Heartbeat(request)
            
            return response.continue_training
            
        except grpc.RpcError as e:
            logger.error(f"Heartbeat failed: {e}")
            return False
    
    def run_training(self, data_dir: str = './data'):
        """Main training loop"""
        logger.info("=" * 60)
        logger.info(f"Starting Federated Learning Client: {self.client_id}")
        logger.info("=" * 60)
        
        # Register with coordinator
        if not self.register():
            logger.error("Registration failed. Exiting.")
            return
        
        # Get training configuration
        if not self.get_training_config():
            logger.error("Failed to get training config. Exiting.")
            return
        
        # Load data
        train_loader, test_loader = self.load_data(data_dir)
        
        # Training loop
        for round_num in range(self.config.total_rounds):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Round {round_num + 1}/{self.config.total_rounds}")
            logger.info(f"{'=' * 60}")
            
            # Check if should continue
            if not self.send_heartbeat(round_num, is_training=False):
                logger.info("Coordinator signaled to stop training")
                break
            
            # Download global model
            self.get_global_model(round_num)
            
            # Train locally
            logger.info("Starting local training...")
            loss, accuracy = self.train_local_model(
                train_loader, 
                self.config.local_epochs
            )
            
            # Submit gradients
            logger.info("Submitting gradients to coordinator...")
            self.submit_gradients(
                round_num, 
                loss, 
                accuracy, 
                len(train_loader.dataset)
            )
            
            # Send heartbeat
            self.send_heartbeat(round_num, is_training=False)
            
            time.sleep(1)  # Small delay between rounds
        
        logger.info("\n" + "=" * 60)
        logger.info("Training complete!")
        logger.info("=" * 60)
        
        # Close connection
        self.channel.close()


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('--client-id', type=str, required=True,
                       help='Unique client identifier')
    parser.add_argument('--server', type=str, default='localhost:50051',
                       help='Coordinator server address')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for MNIST data')
    parser.add_argument('--no-dp', action='store_true',
                       help='Disable differential privacy')
    
    args = parser.parse_args()
    
    # Create and run client
    client = FederatedClient(
        client_id=args.client_id,
        server_address=args.server,
        enable_dp=not args.no_dp
    )
    
    client.run_training(data_dir=args.data_dir)


if __name__ == '__main__':
    main()
