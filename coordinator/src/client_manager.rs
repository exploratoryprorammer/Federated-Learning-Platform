use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{info, warn, debug};

use crate::federated::ClientCapabilities;

/// Information about a registered client
#[derive(Clone, Debug)]
pub struct ClientInfo {
    /// Unique client identifier
    pub client_id: String,
    
    /// Session token for authentication
    pub session_token: String,
    
    /// Assigned client index (0, 1, 2, ...)
    pub client_index: i32,
    
    /// Client capabilities (hardware, data size, etc.)
    pub capabilities: Option<ClientCapabilities>,
    
    /// Last heartbeat timestamp (Unix timestamp)
    pub last_heartbeat: u64,
    
    /// Whether client is currently active
    pub active: bool,
    
    /// Registration timestamp
    pub registered_at: u64,
    
    /// Total number of rounds participated
    pub rounds_participated: i32,
    
    /// Total gradients submitted
    pub gradients_submitted: i32,
}

impl ClientInfo {
    /// Create new client info
    pub fn new(
        client_id: String,
        session_token: String,
        client_index: i32,
        capabilities: Option<ClientCapabilities>,
    ) -> Self {
        let now = Self::current_timestamp();
        
        Self {
            client_id,
            session_token,
            client_index,
            capabilities,
            last_heartbeat: now,
            active: true,
            registered_at: now,
            rounds_participated: 0,
            gradients_submitted: 0,
        }
    }
    
    /// Get current Unix timestamp
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
    
    /// Update heartbeat timestamp
    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Self::current_timestamp();
    }
    
    /// Check if client is inactive (hasn't sent heartbeat in timeout period)
    pub fn is_inactive(&self, timeout_secs: u64) -> bool {
        let now = Self::current_timestamp();
        now - self.last_heartbeat > timeout_secs
    }
    
    /// Mark client as inactive
    pub fn deactivate(&mut self) {
        self.active = false;
    }
    
    /// Increment participation counter
    pub fn increment_participation(&mut self) {
        self.rounds_participated += 1;
    }
    
    /// Increment gradient submission counter
    pub fn increment_submissions(&mut self) {
        self.gradients_submitted += 1;
    }
}

/// Manages all registered clients
pub struct ClientManager {
    /// Map of client_id -> ClientInfo
    clients: HashMap<String, ClientInfo>,
    
    /// Next available client index
    next_index: i32,
    
    /// Total number of registrations (including deactivated)
    total_registrations: usize,
    
    /// Maximum number of clients allowed
    max_clients: usize,
}

impl ClientManager {
    /// Create new client manager
    pub fn new() -> Self {
        Self {
            clients: HashMap::new(),
            next_index: 0,
            total_registrations: 0,
            max_clients: 100, // Default limit
        }
    }
    
    /// Create client manager with custom max clients
    pub fn with_capacity(max_clients: usize) -> Self {
        Self {
            clients: HashMap::new(),
            next_index: 0,
            total_registrations: 0,
            max_clients,
        }
    }
    
    /// Register a new client
    pub fn register(
        &mut self,
        client_id: String,
        session_token: String,
        capabilities: Option<ClientCapabilities>,
    ) -> Result<i32> {
        // Check if already registered
        if self.clients.contains_key(&client_id) {
            warn!("Client {} attempted to register twice", client_id);
            return Err(anyhow!("Client already registered"));
        }
        
        // Check max clients limit
        if self.clients.len() >= self.max_clients {
            warn!(
                "Registration rejected: max clients ({}) reached",
                self.max_clients
            );
            return Err(anyhow!("Maximum number of clients reached"));
        }
        
        // Assign index
        let client_index = self.next_index;
        self.next_index += 1;
        
        // Create client info
        let client_info = ClientInfo::new(
            client_id.clone(),
            session_token,
            client_index,
            capabilities,
        );
        
        // Store client
        self.clients.insert(client_id.clone(), client_info);
        self.total_registrations += 1;
        
        info!(
            "Client {} registered with index {}. Total active: {}",
            client_id,
            client_index,
            self.get_active_count()
        );
        
        Ok(client_index)
    }
    
    /// Validate client session token
    pub fn validate_session(&self, client_id: &str, token: &str) -> bool {
        if let Some(client) = self.clients.get(client_id) {
            client.session_token == token && client.active
        } else {
            false
        }
    }
    
    /// Update client heartbeat
    pub fn update_heartbeat(&mut self, client_id: &str) -> Result<()> {
        if let Some(client) = self.clients.get_mut(client_id) {
            client.update_heartbeat();
            debug!("Heartbeat updated for client {}", client_id);
            Ok(())
        } else {
            Err(anyhow!("Client not found: {}", client_id))
        }
    }
    
    /// Get client info
    pub fn get_client(&self, client_id: &str) -> Option<&ClientInfo> {
        self.clients.get(client_id)
    }
    
    /// Get mutable client info
    pub fn get_client_mut(&mut self, client_id: &str) -> Option<&mut ClientInfo> {
        self.clients.get_mut(client_id)
    }
    
    /// Get count of active clients
    pub fn get_active_count(&self) -> usize {
        self.clients.values().filter(|c| c.active).count()
    }
    
    /// Get total number of clients (including inactive)
    pub fn get_total_count(&self) -> usize {
        self.clients.len()
    }
    
    /// Get all active clients
    pub fn get_active_clients(&self) -> Vec<&ClientInfo> {
        self.clients
            .values()
            .filter(|c| c.active)
            .collect()
    }
    
    /// Get all clients (including inactive)
    pub fn get_all_clients(&self) -> Vec<&ClientInfo> {
        self.clients.values().collect()
    }
    
    /// Cleanup inactive clients based on timeout
    pub fn cleanup_inactive(&mut self, timeout_secs: u64) {
        let mut deactivated = Vec::new();
        
        for (client_id, client) in self.clients.iter_mut() {
            if client.active && client.is_inactive(timeout_secs) {
                client.deactivate();
                deactivated.push(client_id.clone());
            }
        }
        
        if !deactivated.is_empty() {
            warn!(
                "Deactivated {} inactive clients: {:?}",
                deactivated.len(),
                deactivated
            );
        }
    }
    
    /// Remove a client completely
    pub fn remove_client(&mut self, client_id: &str) -> Result<()> {
        if self.clients.remove(client_id).is_some() {
            info!("Client {} removed", client_id);
            Ok(())
        } else {
            Err(anyhow!("Client not found: {}", client_id))
        }
    }
    
    /// Record client participation in a round
    pub fn record_participation(&mut self, client_id: &str) -> Result<()> {
        if let Some(client) = self.clients.get_mut(client_id) {
            client.increment_participation();
            Ok(())
        } else {
            Err(anyhow!("Client not found: {}", client_id))
        }
    }
    
    /// Record gradient submission
    pub fn record_submission(&mut self, client_id: &str) -> Result<()> {
        if let Some(client) = self.clients.get_mut(client_id) {
            client.increment_submissions();
            Ok(())
        } else {
            Err(anyhow!("Client not found: {}", client_id))
        }
    }
    
    /// Get statistics about clients
    pub fn get_stats(&self) -> ClientManagerStats {
        let active_count = self.get_active_count();
        let inactive_count = self.clients.len() - active_count;
        
        let total_participation: i32 = self.clients
            .values()
            .map(|c| c.rounds_participated)
            .sum();
        
        let total_submissions: i32 = self.clients
            .values()
            .map(|c| c.gradients_submitted)
            .sum();
        
        let avg_participation = if !self.clients.is_empty() {
            total_participation as f64 / self.clients.len() as f64
        } else {
            0.0
        };
        
        ClientManagerStats {
            active_clients: active_count,
            inactive_clients: inactive_count,
            total_clients: self.clients.len(),
            total_registrations: self.total_registrations,
            average_participation: avg_participation,
            total_participation,
            total_submissions,
        }
    }
    
    /// Reset all client statistics
    pub fn reset_stats(&mut self) {
        for client in self.clients.values_mut() {
            client.rounds_participated = 0;
            client.gradients_submitted = 0;
        }
    }
    
    /// Clear all clients
    pub fn clear(&mut self) {
        self.clients.clear();
        self.next_index = 0;
        info!("All clients cleared");
    }
}

/// Statistics about client manager
#[derive(Debug, Clone)]
pub struct ClientManagerStats {
    pub active_clients: usize,
    pub inactive_clients: usize,
    pub total_clients: usize,
    pub total_registrations: usize,
    pub average_participation: f64,
    pub total_participation: i32,
    pub total_submissions: i32,
}

impl Default for ClientManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_client_registration() {
        let mut manager = ClientManager::new();
        
        let result = manager.register(
            "client1".to_string(),
            "token123".to_string(),
            None,
        );
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
        assert_eq!(manager.get_active_count(), 1);
    }
    
    #[test]
    fn test_duplicate_registration() {
        let mut manager = ClientManager::new();
        
        manager.register("client1".to_string(), "token1".to_string(), None).unwrap();
        let result = manager.register("client1".to_string(), "token2".to_string(), None);
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_session_validation() {
        let mut manager = ClientManager::new();
        
        manager.register("client1".to_string(), "valid_token".to_string(), None).unwrap();
        
        assert!(manager.validate_session("client1", "valid_token"));
        assert!(!manager.validate_session("client1", "invalid_token"));
        assert!(!manager.validate_session("client2", "valid_token"));
    }
    
    #[test]
    fn test_heartbeat() {
        let mut manager = ClientManager::new();
        
        manager.register("client1".to_string(), "token".to_string(), None).unwrap();
        
        // Get initial heartbeat
        let client1 = manager.get_client("client1").unwrap();
        let initial_heartbeat = client1.last_heartbeat;
        
        // Wait a bit and update
        std::thread::sleep(std::time::Duration::from_millis(100));
        manager.update_heartbeat("client1").unwrap();
        
        // Verify heartbeat was updated
        let client1 = manager.get_client("client1").unwrap();
        assert!(client1.last_heartbeat > initial_heartbeat);
    }
    
    #[test]
    fn test_cleanup_inactive() {
        let mut manager = ClientManager::new();
        
        manager.register("client1".to_string(), "token1".to_string(), None).unwrap();
        
        // Manually set old heartbeat
        if let Some(client) = manager.get_client_mut("client1") {
            client.last_heartbeat = 0; // Very old timestamp
        }
        
        // Cleanup with 1 second timeout
        manager.cleanup_inactive(1);
        
        // Client should be inactive
        let client1 = manager.get_client("client1").unwrap();
        assert!(!client1.active);
        assert_eq!(manager.get_active_count(), 0);
    }
    
    #[test]
    fn test_participation_tracking() {
        let mut manager = ClientManager::new();
        
        manager.register("client1".to_string(), "token".to_string(), None).unwrap();
        
        manager.record_participation("client1").unwrap();
        manager.record_participation("client1").unwrap();
        manager.record_submission("client1").unwrap();
        
        let client = manager.get_client("client1").unwrap();
        assert_eq!(client.rounds_participated, 2);
        assert_eq!(client.gradients_submitted, 1);
    }
    
    #[test]
    fn test_stats() {
        let mut manager = ClientManager::new();
        
        manager.register("client1".to_string(), "token1".to_string(), None).unwrap();
        manager.register("client2".to_string(), "token2".to_string(), None).unwrap();
        
        manager.record_participation("client1").unwrap();
        manager.record_participation("client2").unwrap();
        manager.record_participation("client2").unwrap();
        
        let stats = manager.get_stats();
        assert_eq!(stats.active_clients, 2);
        assert_eq!(stats.total_participation, 3);
        assert_eq!(stats.average_participation, 1.5);
    }
    
    #[test]
    fn test_max_clients_limit() {
        let mut manager = ClientManager::with_capacity(2);
        
        manager.register("client1".to_string(), "token1".to_string(), None).unwrap();
        manager.register("client2".to_string(), "token2".to_string(), None).unwrap();
        
        let result = manager.register("client3".to_string(), "token3".to_string(), None);
        assert!(result.is_err());
    }
}