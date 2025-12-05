// rust-server/src/coordinator.rs

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;
use tracing::{info, warn, debug};

use crate::federated::*;
use crate::client_manager::ClientManager;

/// Main coordinator state managing federated learning rounds
pub struct CoordinatorState {
    /// Current training round number
    current_round: i32,
    
    /// Total number of rounds to complete
    total_rounds: i32,
    
    /// Global model weights (serialized)
    global_model: Vec<u8>,
    
    /// Model version counter
    model_version: i64,
    
    /// Client manager for registration and tracking
    client_manager: ClientManager,
    
    /// Buffer storing gradients per round
    gradients_buffer: HashMap<i32, Vec<GradientUpdate>>,
    
    /// Training configuration
    config: TrainingConfig,
    
    /// Minimum number of clients needed to trigger aggregation
    aggregation_threshold: i32,
    
    /// Track which clients participated in each round
    round_participants: HashMap<i32, Vec<String>>,
    
    /// Store aggregation results per round
    round_results: HashMap<i32, RoundResult>,
}

/// Results from a completed training round
#[derive(Clone, Debug)]
pub struct RoundResult {
    pub round_number: i32,
    pub num_clients: i32,
    pub avg_loss: f32,
    pub avg_accuracy: f32,
    pub aggregation_time_ms: f32,
    pub byzantine_detected: i32,
}

impl CoordinatorState {
    /// Create a new coordinator with default configuration
    pub fn new() -> Self {
        info!("Initializing coordinator state");
        
        Self {
            current_round: 0,
            total_rounds: 10,
            global_model: Self::initialize_model(),
            model_version: 0,
            client_manager: ClientManager::new(),
            gradients_buffer: HashMap::new(),
            config: Self::default_config(),
            aggregation_threshold: 3,
            round_participants: HashMap::new(),
            round_results: HashMap::new(),
        }
    }
    
    /// Initialize a dummy model (in production, load actual model weights)
    fn initialize_model() -> Vec<u8> {
        // Create dummy model weights
        // In production: load from file or initialize PyTorch model
        info!("Initializing global model");
        vec![0u8; 1024] // Placeholder
    }
    
    /// Create default training configuration
    fn default_config() -> TrainingConfig {
        TrainingConfig {
            total_rounds: 10,
            local_epochs: 5,
            batch_size: 32,
            learning_rate: 0.01,
            enable_compression: true,
            enable_differential_privacy: true,
            dp_config: Some(DpParameters {
                epsilon: 2.0,
                delta: 1e-5,
                noise_multiplier: 1.1,
                max_grad_norm: 1.0,
            }),
            aggregation_threshold: 3,
        }
    }
    
    /// Register a new client
    pub async fn register_client(
        &mut self,
        registration: ClientRegistration,
    ) -> Result<ClientRegistrationResponse> {
        info!("Processing registration for client: {}", registration.client_id);
        
        // Generate session token
        let session_token = Uuid::new_v4().to_string();
        
        // Register with client manager
        let client_index = self.client_manager.register(
            registration.client_id.clone(),
            session_token.clone(),
            registration.capabilities,
        )?;
        
        info!(
            "Client {} registered with index {} and token {}",
            registration.client_id,
            client_index,
            &session_token[..8] // Log only first 8 chars of token
        );
        
        Ok(ClientRegistrationResponse {
            success: true,
            session_token,
            assigned_client_index: client_index,
            message: format!(
                "Successfully registered as client {}. Ready for training.",
                client_index
            ),
        })
    }
    
    /// Validate client session
    pub fn validate_session(&self, client_id: &str, token: &str) -> bool {
        self.client_manager.validate_session(client_id, token)
    }
    
    /// Get current global model for a client
    pub async fn get_global_model(&self, requested_round: i32) -> Result<ModelResponse> {
        debug!(
            "Serving global model for round {} (current: {})",
            requested_round, self.current_round
        );
        
        // In production, you would:
        // 1. Check if client is requesting the right round
        // 2. Load actual model weights from storage
        // 3. Serialize PyTorch/TensorFlow model
        
        Ok(ModelResponse {
            round_number: self.current_round,
            model_weights: self.global_model.clone(),
            model_version: self.model_version,
            metadata: Some(ModelMetadata {
                total_parameters: self.global_model.len() as i32,
                model_architecture: "SimpleCNN".to_string(),
                learning_rate: self.config.learning_rate,
                batch_size: self.config.batch_size,
            }),
        })
    }
    
    /// Submit gradients from a client
    pub async fn submit_gradients(&mut self, update: GradientUpdate) -> Result<GradientAck> {
        let client_id = update.client_id.clone();
        let round = update.round_number;
        
        debug!(
            "Received gradients from {} for round {}",
            client_id, round
        );
        
        // Validate round number
        if round != self.current_round {
            warn!(
                "Client {} submitted gradients for wrong round (expected {}, got {})",
                client_id, self.current_round, round
            );
            return Ok(GradientAck {
                accepted: false,
                message: format!(
                    "Wrong round. Expected {}, got {}",
                    self.current_round, round
                ),
                next_round: self.current_round,
            });
        }
        
        // Check if client already submitted for this round
        if let Some(participants) = self.round_participants.get(&round) {
            if participants.contains(&client_id) {
                warn!("Client {} already submitted for round {}", client_id, round);
                return Ok(GradientAck {
                    accepted: false,
                    message: "Already submitted for this round".to_string(),
                    next_round: self.current_round,
                });
            }
        }
        
        // Add to buffer
        self.gradients_buffer
            .entry(round)
            .or_insert_with(Vec::new)
            .push(update);
        
        // Track participant
        self.round_participants
            .entry(round)
            .or_insert_with(Vec::new)
            .push(client_id.clone());
        
        // Check if we have enough gradients for aggregation
        let gradient_count = self.gradients_buffer.get(&round).map(|v| v.len()).unwrap_or(0);
        
        info!(
            "Gradient count for round {}: {}/{}",
            round, gradient_count, self.aggregation_threshold
        );
        
        if gradient_count >= self.aggregation_threshold as usize {
            info!("Threshold reached. Triggering aggregation for round {}", round);
            self.trigger_aggregation(round).await?;
        }
        
        Ok(GradientAck {
            accepted: true,
            message: format!(
                "Gradient received ({}/{})",
                gradient_count, self.aggregation_threshold
            ),
            next_round: self.current_round,
        })
    }
    
    /// Trigger gradient aggregation when threshold is reached
    async fn trigger_aggregation(&mut self, round: i32) -> Result<()> {
        let start_time = SystemTime::now();
        
        if let Some(gradients) = self.gradients_buffer.remove(&round) {
            info!(
                "Aggregating gradients for round {} from {} clients",
                round,
                gradients.len()
            );
            
            // Calculate average loss and accuracy
            let total_loss: f32 = gradients
                .iter()
                .filter_map(|g| g.metadata.as_ref().map(|m| m.local_loss))
                .sum();
            let total_accuracy: f32 = gradients
                .iter()
                .filter_map(|g| g.metadata.as_ref().map(|m| m.local_accuracy))
                .sum();
            
            let num_clients = gradients.len() as f32;
            let avg_loss = total_loss / num_clients;
            let avg_accuracy = total_accuracy / num_clients;
            
            // In production: Call C++ aggregator via gRPC
            // For now, simulate aggregation
            // let aggregated = self.call_cpp_aggregator(gradients).await?;
            
            // Simulate Byzantine detection (in production, comes from C++ aggregator)
            let byzantine_detected = 0;
            
            // Update model version and advance round
            self.model_version += 1;
            self.current_round += 1;
            
            let elapsed = start_time.elapsed()?.as_millis() as f32;
            
            // Store round result
            let result = RoundResult {
                round_number: round,
                num_clients: num_clients as i32,
                avg_loss,
                avg_accuracy,
                aggregation_time_ms: elapsed,
                byzantine_detected,
            };
            
            self.round_results.insert(round, result.clone());
            
            info!(
                "Round {} complete: loss={:.4}, accuracy={:.2}%, time={:.2}ms",
                round, avg_loss, avg_accuracy, elapsed
            );
            
            // Update metrics
            crate::metrics::update_round_metrics(self.current_round, avg_accuracy as f64);
            crate::metrics::record_aggregation(
                elapsed as f64 / 1000.0,
                0.75, // compression ratio
                byzantine_detected,
            );
        }
        
        Ok(())
    }
    
    /// Process heartbeat from client
    pub async fn process_heartbeat(
        &mut self,
        request: HeartbeatRequest,
    ) -> Result<HeartbeatResponse> {
        debug!("Heartbeat from {}", request.client_id);
        
        // Update client heartbeat timestamp
        self.client_manager.update_heartbeat(&request.client_id)?;
        
        // Check if training should continue
        let continue_training = self.current_round < self.total_rounds;
        
        Ok(HeartbeatResponse {
            continue_training,
            next_round: self.current_round,
            message: if continue_training {
                format!("Training in progress (round {}/{})", self.current_round, self.total_rounds)
            } else {
                "Training complete".to_string()
            },
        })
    }
    
    /// Get training configuration
    pub fn get_config(&self) -> TrainingConfig {
        self.config.clone()
    }
    
    /// Get current round number
    pub fn get_current_round(&self) -> i32 {
        self.current_round
    }
    
    /// Get active client count
    pub fn get_active_client_count(&self) -> usize {
        self.client_manager.get_active_count()
    }
    
    /// Get round results for monitoring
    pub fn get_round_results(&self) -> Vec<RoundResult> {
        let mut results: Vec<_> = self.round_results.values().cloned().collect();
        results.sort_by_key(|r| r.round_number);
        results
    }
    
    /// Cleanup inactive clients (should be called periodically)
    pub fn cleanup_inactive_clients(&mut self, timeout_secs: u64) {
        self.client_manager.cleanup_inactive(timeout_secs);
        
        // Update metrics
        crate::metrics::ACTIVE_CLIENTS.set(self.get_active_client_count() as f64);
    }
    
    /// Reset coordinator state for new training run
    pub fn reset(&mut self) {
        info!("Resetting coordinator state");
        self.current_round = 0;
        self.model_version = 0;
        self.global_model = Self::initialize_model();
        self.gradients_buffer.clear();
        self.round_participants.clear();
        self.round_results.clear();
    }
}

impl Default for CoordinatorState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_client_registration() {
        let mut coordinator = CoordinatorState::new();
        
        let registration = ClientRegistration {
            client_id: "test_client".to_string(),
            client_version: "1.0.0".to_string(),
            capabilities: None,
            timestamp: 0,
        };
        
        let response = coordinator.register_client(registration).await.unwrap();
        
        assert!(response.success);
        assert!(!response.session_token.is_empty());
        assert_eq!(response.assigned_client_index, 0);
    }
    
    #[tokio::test]
    async fn test_gradient_submission() {
        let mut coordinator = CoordinatorState::new();
        coordinator.aggregation_threshold = 2; // Lower threshold for testing
        
        // Register clients
        let reg1 = ClientRegistration {
            client_id: "client1".to_string(),
            client_version: "1.0".to_string(),
            capabilities: None,
            timestamp: 0,
        };
        let resp1 = coordinator.register_client(reg1).await.unwrap();
        
        // Submit gradient
        let gradient = GradientUpdate {
            client_id: "client1".to_string(),
            session_token: resp1.session_token,
            round_number: 0,
            gradients: vec![1, 2, 3, 4],
            metadata: Some(GradientMetadata {
                num_samples_trained: 100,
                local_loss: 0.5,
                local_accuracy: 90.0,
                is_compressed: false,
                compression_ratio: 1.0,
                has_differential_privacy: false,
                dp_params: None,
            }),
            timestamp: 0,
        };
        
        let ack = coordinator.submit_gradients(gradient).await.unwrap();
        assert!(ack.accepted);
    }
}