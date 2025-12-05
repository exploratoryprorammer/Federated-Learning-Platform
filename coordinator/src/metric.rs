use lazy_static::lazy_static;
use prometheus::{
    register_counter, register_counter_vec, register_gauge, register_gauge_vec,
    register_histogram, register_histogram_vec, Counter, CounterVec, Encoder, Gauge, GaugeVec,
    Histogram, HistogramVec, TextEncoder,
};
use std::net::SocketAddr;
use tracing::{info, error};
use warp::Filter;

// Define all Prometheus metrics using lazy_static
lazy_static! {
    /// Number of currently active clients
    pub static ref ACTIVE_CLIENTS: Gauge =
        register_gauge!(
            "federated_active_clients",
            "Number of active clients connected to the coordinator"
        ).unwrap();
    
    /// Total number of client registrations
    pub static ref TOTAL_REGISTRATIONS: Counter =
        register_counter!(
            "federated_total_registrations",
            "Total number of client registration attempts"
        ).unwrap();
    
    /// Total model requests served
    pub static ref MODEL_REQUESTS: Counter =
        register_counter!(
            "federated_model_requests_total",
            "Total number of global model requests from clients"
        ).unwrap();
    
    /// Total gradients received from clients
    pub static ref GRADIENTS_RECEIVED: Counter =
        register_counter!(
            "federated_gradients_received_total",
            "Total number of gradient updates received"
        ).unwrap();
    
    /// Total heartbeats received
    pub static ref HEARTBEATS_RECEIVED: Counter =
        register_counter!(
            "federated_heartbeats_received_total",
            "Total number of heartbeat messages received"
        ).unwrap();
    
    /// Current training round number
    pub static ref CURRENT_ROUND: Gauge =
        register_gauge!(
            "federated_current_round",
            "Current federated learning training round"
        ).unwrap();
    
    /// Time taken to submit gradients (client -> server)
    pub static ref GRADIENT_SUBMISSION_TIME: Histogram = register_histogram!(
        "federated_gradient_submission_seconds",
        "Time taken for gradient submission from client to server"
    ).unwrap();
    
    /// Time taken for gradient aggregation
    pub static ref AGGREGATION_TIME: Histogram = register_histogram!(
        "federated_aggregation_seconds",
        "Time taken to aggregate gradients from multiple clients",
        vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    ).unwrap();
    
    /// Global model accuracy
    pub static ref MODEL_ACCURACY: Gauge =
        register_gauge!(
            "federated_model_accuracy",
            "Global model accuracy (0.0 to 1.0)"
        ).unwrap();
    
    /// Number of Byzantine clients detected
    pub static ref BYZANTINE_CLIENTS_DETECTED: Counter =
        register_counter!(
            "federated_byzantine_clients_detected_total",
            "Total number of Byzantine (malicious) clients detected"
        ).unwrap();
    
    /// Average gradient compression ratio
    pub static ref COMPRESSION_RATIO: Gauge =
        register_gauge!(
            "federated_compression_ratio",
            "Average gradient compression ratio (0.0 to 1.0)"
        ).unwrap();
    
    /// Round-specific metrics with labels
    pub static ref ROUND_LOSS: GaugeVec =
        register_gauge_vec!(
            "federated_round_loss",
            "Training loss per round",
            &["round"]
        ).unwrap();
    
    pub static ref ROUND_ACCURACY: GaugeVec =
        register_gauge_vec!(
            "federated_round_accuracy",
            "Training accuracy per round",
            &["round"]
        ).unwrap();
    
    pub static ref ROUND_CLIENTS: GaugeVec =
        register_gauge_vec!(
            "federated_round_clients",
            "Number of clients participated in round",
            &["round"]
        ).unwrap();
    
    /// Client-specific metrics
    pub static ref CLIENT_PARTICIPATION: CounterVec =
        register_counter_vec!(
            "federated_client_participation_total",
            "Number of rounds each client participated in",
            &["client_id"]
        ).unwrap();
    
    pub static ref CLIENT_GRADIENTS: CounterVec =
        register_counter_vec!(
            "federated_client_gradients_total",
            "Number of gradient submissions per client",
            &["client_id"]
        ).unwrap();
    
    /// gRPC request metrics
    pub static ref GRPC_REQUESTS: CounterVec =
        register_counter_vec!(
            "federated_grpc_requests_total",
            "Total gRPC requests by method",
            &["method"]
        ).unwrap();
    
    pub static ref GRPC_REQUEST_DURATION: HistogramVec =
        register_histogram_vec!(
            "federated_grpc_request_duration_seconds",
            "gRPC request duration by method",
            &["method"]
        ).unwrap();
    
    pub static ref GRPC_ERRORS: CounterVec =
        register_counter_vec!(
            "federated_grpc_errors_total",
            "Total gRPC errors by method and status",
            &["method", "status"]
        ).unwrap();
}

/// Record gradient submission timing
pub fn record_gradient_submission() {
    // In production, measure actual time
    GRADIENT_SUBMISSION_TIME.observe(0.05); // Placeholder
}

/// Update metrics for completed training round
pub fn update_round_metrics(round: i32, accuracy: f64) {
    CURRENT_ROUND.set(round as f64);
    MODEL_ACCURACY.set(accuracy);
    
    // Update round-specific metrics
    let round_label = round.to_string();
    ROUND_ACCURACY.with_label_values(&[&round_label]).set(accuracy);
    
    info!(
        "Updated metrics for round {}: accuracy={:.4}",
        round, accuracy
    );
}

/// Record aggregation completion with statistics
pub fn record_aggregation(
    duration_secs: f64,
    compression_ratio: f64,
    byzantine_count: u64,
) {
    AGGREGATION_TIME.observe(duration_secs);
    COMPRESSION_RATIO.set(compression_ratio);
    BYZANTINE_CLIENTS_DETECTED.inc_by(byzantine_count);
    
    info!(
        "Aggregation metrics: duration={:.3}s, compression={:.2}%, byzantine={}",
        duration_secs,
        compression_ratio * 100.0,
        byzantine_count
    );
}

/// Record metrics for a specific round
pub fn record_round_completion(
    round: i32,
    num_clients: usize,
    avg_loss: f32,
    avg_accuracy: f32,
) {
    let round_label = round.to_string();
    
    ROUND_LOSS.with_label_values(&[&round_label]).set(avg_loss as f64);
    ROUND_ACCURACY.with_label_values(&[&round_label]).set(avg_accuracy as f64);
    ROUND_CLIENTS.with_label_values(&[&round_label]).set(num_clients as f64);
    
    info!(
        "Round {} completed: {} clients, loss={:.4}, accuracy={:.2}%",
        round, num_clients, avg_loss, avg_accuracy
    );
}

/// Record client participation
pub fn record_client_participation(client_id: &str) {
    CLIENT_PARTICIPATION.with_label_values(&[client_id]).inc();
}

/// Record client gradient submission
pub fn record_client_gradient(client_id: &str) {
    CLIENT_GRADIENTS.with_label_values(&[client_id]).inc();
}

/// Record gRPC method call
pub fn record_grpc_call(method: &str, duration_secs: f64) {
    GRPC_REQUESTS.with_label_values(&[method]).inc();
    GRPC_REQUEST_DURATION.with_label_values(&[method]).observe(duration_secs);
}

/// Record gRPC error
pub fn record_grpc_error(method: &str, status: &str) {
    GRPC_ERRORS.with_label_values(&[method, status]).inc();
}

/// Setup Prometheus metrics HTTP server
pub async fn setup_metrics() -> anyhow::Result<()> {
    // Create metrics endpoint
    let metrics_route = warp::path!("metrics").and_then(metrics_handler);
    
    // Create health check endpoint
    let health_route = warp::path!("health").map(|| {
        warp::reply::json(&serde_json::json!({
            "status": "healthy",
            "service": "federated-coordinator"
        }))
    });
    
    // Combine routes
    let routes = metrics_route.or(health_route);

    let addr: SocketAddr = "0.0.0.0:9090".parse()?;
    info!("Metrics server listening on: http://{}/metrics", addr);
    info!("Health check available at: http://{}/health", addr);

    warp::serve(routes).run(addr).await;

    Ok(())
}

/// Handler for /metrics endpoint
async fn metrics_handler() -> Result<impl warp::Reply, warp::Rejection> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = vec![];
    
    match encoder.encode(&metric_families, &mut buffer) {
        Ok(_) => {
            Ok(warp::reply::with_header(
                buffer,
                "Content-Type",
                encoder.format_type(),
            ))
        }
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            Err(warp::reject::reject())
        }
    }
}

/// Get current metrics snapshot (for debugging)
pub fn get_metrics_snapshot() -> MetricsSnapshot {
    MetricsSnapshot {
        active_clients: ACTIVE_CLIENTS.get(),
        total_registrations: TOTAL_REGISTRATIONS.get(),
        model_requests: MODEL_REQUESTS.get(),
        gradients_received: GRADIENTS_RECEIVED.get(),
        heartbeats_received: HEARTBEATS_RECEIVED.get(),
        current_round: CURRENT_ROUND.get(),
        model_accuracy: MODEL_ACCURACY.get(),
        byzantine_detected: BYZANTINE_CLIENTS_DETECTED.get(),
        compression_ratio: COMPRESSION_RATIO.get(),
    }
}

/// Snapshot of current metrics
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub active_clients: f64,
    pub total_registrations: f64,
    pub model_requests: f64,
    pub gradients_received: f64,
    pub heartbeats_received: f64,
    pub current_round: f64,
    pub model_accuracy: f64,
    pub byzantine_detected: f64,
    pub compression_ratio: f64,
}

impl std::fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Metrics Snapshot:\n\
             - Active Clients: {}\n\
             - Total Registrations: {}\n\
             - Model Requests: {}\n\
             - Gradients Received: {}\n\
             - Heartbeats: {}\n\
             - Current Round: {}\n\
             - Model Accuracy: {:.2}%\n\
             - Byzantine Detected: {}\n\
             - Compression Ratio: {:.2}%",
            self.active_clients,
            self.total_registrations,
            self.model_requests,
            self.gradients_received,
            self.heartbeats_received,
            self.current_round,
            self.model_accuracy * 100.0,
            self.byzantine_detected,
            self.compression_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_increment() {
        let initial = TOTAL_REGISTRATIONS.get();
        TOTAL_REGISTRATIONS.inc();
        assert_eq!(TOTAL_REGISTRATIONS.get(), initial + 1.0);
    }
    
    #[test]
    fn test_gauge_set() {
        ACTIVE_CLIENTS.set(5.0);
        assert_eq!(ACTIVE_CLIENTS.get(), 5.0);
    }
    
    #[test]
    fn test_metrics_snapshot() {
        let snapshot = get_metrics_snapshot();
        assert!(snapshot.active_clients >= 0.0);
        assert!(snapshot.model_accuracy >= 0.0 && snapshot.model_accuracy <= 1.0);
    }
}