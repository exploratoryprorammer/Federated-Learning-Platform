use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, warn};

mod coordinator;
mod client_manager;
mod metrics;

use coordinator::CoordinatorState;
use metrics::setup_metrics;

// Include generated protobuf code
pub mod federated {
    tonic::include_proto!("federated");
}

use federated::federated_coordinator_server::{FederatedCoordinator, FederatedCoordinatorServer};
use federated::*;

pub struct FederatedCoordinatorService {
    state: Arc<RwLock<CoordinatorState>>,
}

impl FederatedCoordinatorService {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(CoordinatorState::new())),
        }
    }
}

#[tonic::async_trait]
impl FederatedCoordinator for FederatedCoordinatorService {
    async fn register_client(
        &self,
        request: Request<ClientRegistration>,
    ) -> Result<Response<ClientRegistrationResponse>, Status> {
        let req = request.into_inner();
        info!("Registering client: {}", req.client_id);

        let mut state = self.state.write().await;
        
        match state.register_client(req).await {
            Ok(response) => {
                metrics::ACTIVE_CLIENTS.inc();
                metrics::TOTAL_REGISTRATIONS.inc();
                Ok(Response::new(response))
            }
            Err(e) => {
                warn!("Client registration failed: {}", e);
                Err(Status::internal(format!("Registration failed: {}", e)))
            }
        }
    }

    async fn get_global_model(
        &self,
        request: Request<ModelRequest>,
    ) -> Result<Response<ModelResponse>, Status> {
        let req = request.into_inner();
        let state = self.state.read().await;

        if !state.validate_session(&req.client_id, &req.session_token) {
            return Err(Status::unauthenticated("Invalid session"));
        }

        match state.get_global_model(req.current_round).await {
            Ok(model) => {
                metrics::MODEL_REQUESTS.inc();
                Ok(Response::new(model))
            }
            Err(e) => Err(Status::internal(format!("Model retrieval failed: {}", e))),
        }
    }

    async fn submit_gradients(
        &self,
        request: Request<GradientUpdate>,
    ) -> Result<Response<GradientAck>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;

        if !state.validate_session(&req.client_id, &req.session_token) {
            return Err(Status::unauthenticated("Invalid session"));
        }

        match state.submit_gradients(req).await {
            Ok(ack) => {
                metrics::GRADIENTS_RECEIVED.inc();
                metrics::record_gradient_submission();
                Ok(Response::new(ack))
            }
            Err(e) => Err(Status::internal(format!("Gradient submission failed: {}", e))),
        }
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        let mut state = self.state.write().await;

        if !state.validate_session(&req.client_id, &req.session_token) {
            return Err(Status::unauthenticated("Invalid session"));
        }

        match state.process_heartbeat(req).await {
            Ok(response) => {
                metrics::HEARTBEATS_RECEIVED.inc();
                Ok(Response::new(response))
            }
            Err(e) => Err(Status::internal(format!("Heartbeat failed: {}", e))),
        }
    }

    async fn get_training_config(
        &self,
        request: Request<ConfigRequest>,
    ) -> Result<Response<TrainingConfig>, Status> {
        let _req = request.into_inner();
        let state = self.state.read().await;

        Ok(Response::new(state.get_config()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    info!("Starting Federated Learning Coordinator");

    // Setup Prometheus metrics endpoint
    let metrics_handle = tokio::spawn(setup_metrics());

    // Create gRPC service
    let addr = "0.0.0.0:50051".parse()?;
    let service = FederatedCoordinatorService::new();

    info!("Coordinator listening on {}", addr);

    // Start gRPC server
    Server::builder()
        .add_service(FederatedCoordinatorServer::new(service))
        .serve(addr)
        .await?;

    metrics_handle.await??;

    Ok(())
}