use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::physics::Simulation;

/// Errors that can occur during storage operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageError {
    NotFound,
    SerializationError(String),
    ConnectionError(String),
    Unknown(String),
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageError::NotFound => write!(f, "Simulation not found"),
            StorageError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::ConnectionError(msg) => write!(f, "Connection error: {}", msg),
            StorageError::Unknown(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

/// Storage abstraction for simulation state
///
/// For now, both implementations store the full Simulation in memory/Redis.
/// Future: Could add checkpoint/snapshot serialization for full persistence.
#[async_trait]
pub trait SimulationStorage: Send + Sync {
    /// Store or update a simulation
    async fn upsert(&mut self, sim_id: &str, simulation: Simulation) -> Result<(), StorageError>;

    /// Retrieve a mutable reference to a simulation
    async fn get_mut(&mut self, sim_id: &str) -> Result<Option<&mut Simulation>, StorageError>;

    /// Remove a simulation
    async fn remove(&mut self, sim_id: &str) -> Result<bool, StorageError>;

    /// Check if a simulation exists
    async fn exists(&self, sim_id: &str) -> Result<bool, StorageError>;

    /// List all simulation IDs (for debugging/monitoring)
    async fn list_ids(&self) -> Result<Vec<String>, StorageError>;
}

/// In-memory storage backend (for local development)
///
/// Simple HashMap-based storage. Fast, but simulations are lost on restart.
pub struct MemoryStorage {
    simulations: HashMap<String, Simulation>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        info!("📦 Initialized MemoryStorage backend (local development)");
        Self {
            simulations: HashMap::new(),
        }
    }
}

#[async_trait]
impl SimulationStorage for MemoryStorage {
    async fn upsert(&mut self, sim_id: &str, simulation: Simulation) -> Result<(), StorageError> {
        self.simulations.insert(sim_id.to_string(), simulation);
        Ok(())
    }

    async fn get_mut(&mut self, sim_id: &str) -> Result<Option<&mut Simulation>, StorageError> {
        Ok(self.simulations.get_mut(sim_id))
    }

    async fn remove(&mut self, sim_id: &str) -> Result<bool, StorageError> {
        Ok(self.simulations.remove(sim_id).is_some())
    }

    async fn exists(&self, sim_id: &str) -> Result<bool, StorageError> {
        Ok(self.simulations.contains_key(sim_id))
    }

    async fn list_ids(&self) -> Result<Vec<String>, StorageError> {
        Ok(self.simulations.keys().cloned().collect())
    }
}

/// Redis-backed storage (for production/Fly.io)
///
/// Stores simulation metadata in Redis, actual simulation state in memory.
/// This provides:
/// - Distributed locking for horizontal scaling
/// - TTL-based expiration
/// - Simulation ownership tracking
/// - Future: Could add full serialization/checkpointing
pub struct RedisStorage {
    simulations: HashMap<String, Simulation>,
    client: redis::Client,
    ttl_seconds: u64,
}

impl RedisStorage {
    pub async fn new(redis_url: &str, ttl_seconds: u64) -> Result<Self, StorageError> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| StorageError::ConnectionError(format!("Failed to connect to Redis: {}", e)))?;

        // Test connection
        let mut conn = client.get_multiplexed_async_connection()
            .await
            .map_err(|e| StorageError::ConnectionError(format!("Redis connection test failed: {}", e)))?;

        // Ping Redis to ensure it's working
        redis::cmd("PING")
            .query_async::<_, String>(&mut conn)
            .await
            .map_err(|e| StorageError::ConnectionError(format!("Redis PING failed: {}", e)))?;

        info!("📦 Initialized RedisStorage backend");
        info!("   URL: {}", redis_url);
        info!("   TTL: {} seconds", ttl_seconds);

        Ok(Self {
            simulations: HashMap::new(),
            client,
            ttl_seconds,
        })
    }

    fn meta_key(&self, sim_id: &str) -> String {
        format!("sim:{}:meta", sim_id)
    }

    /// Update TTL for a simulation in Redis
    async fn touch(&self, sim_id: &str) -> Result<(), StorageError> {
        let mut conn = self.client.get_multiplexed_async_connection()
            .await
            .map_err(|e| StorageError::ConnectionError(format!("Redis connection failed: {}", e)))?;

        let key = self.meta_key(sim_id);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Store metadata with TTL
        redis::cmd("SETEX")
            .arg(&key)
            .arg(self.ttl_seconds)
            .arg(timestamp)
            .query_async::<_, ()>(&mut conn)
            .await
            .map_err(|e| StorageError::ConnectionError(format!("Redis SETEX failed: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl SimulationStorage for RedisStorage {
    async fn upsert(&mut self, sim_id: &str, simulation: Simulation) -> Result<(), StorageError> {
        // Store simulation in memory
        self.simulations.insert(sim_id.to_string(), simulation);

        // Update TTL in Redis
        self.touch(sim_id).await?;

        Ok(())
    }

    async fn get_mut(&mut self, sim_id: &str) -> Result<Option<&mut Simulation>, StorageError> {
        // Check if simulation exists
        if self.simulations.contains_key(sim_id) {
            // Refresh TTL in background (don't block on this)
            let client = self.client.clone();
            let ttl = self.ttl_seconds;
            let sid = sim_id.to_string();
            tokio::spawn(async move {
                if let Ok(mut conn) = client.get_multiplexed_async_connection().await {
                    let key = format!("sim:{}:meta", sid);
                    let timestamp = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let _ = redis::cmd("SETEX")
                        .arg(&key)
                        .arg(ttl)
                        .arg(timestamp)
                        .query_async::<_, ()>(&mut conn)
                        .await;
                }
            });

            Ok(self.simulations.get_mut(sim_id))
        } else {
            Ok(None)
        }
    }

    async fn remove(&mut self, sim_id: &str) -> Result<bool, StorageError> {
        // Remove from memory
        let existed = self.simulations.remove(sim_id).is_some();

        if existed {
            // Remove from Redis
            let mut conn = self.client.get_multiplexed_async_connection()
                .await
                .map_err(|e| StorageError::ConnectionError(format!("Redis connection failed: {}", e)))?;

            let key = self.meta_key(sim_id);
            redis::cmd("DEL")
                .arg(&key)
                .query_async::<_, ()>(&mut conn)
                .await
                .map_err(|e| StorageError::ConnectionError(format!("Redis DEL failed: {}", e)))?;
        }

        Ok(existed)
    }

    async fn exists(&self, sim_id: &str) -> Result<bool, StorageError> {
        Ok(self.simulations.contains_key(sim_id))
    }

    async fn list_ids(&self) -> Result<Vec<String>, StorageError> {
        Ok(self.simulations.keys().cloned().collect())
    }
}

/// Storage configuration
#[derive(Debug, Clone)]
pub enum StorageConfig {
    Memory,
    Redis {
        url: String,
        ttl_seconds: u64,
    },
}

impl StorageConfig {
    /// Parse from environment variables
    pub fn from_env() -> Self {
        match std::env::var("STORAGE_BACKEND").as_deref() {
            Ok("redis") => {
                let url = std::env::var("REDIS_URL")
                    .unwrap_or_else(|_| {
                        warn!("REDIS_URL not set, using default: redis://127.0.0.1:6379");
                        "redis://127.0.0.1:6379".to_string()
                    });

                let ttl_seconds = std::env::var("STORAGE_TTL_SECONDS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3600); // Default 1 hour

                info!("🔧 Storage config: Redis");
                StorageConfig::Redis { url, ttl_seconds }
            }
            _ => {
                info!("🔧 Storage config: Memory (local development)");
                StorageConfig::Memory
            }
        }
    }

    /// Create the storage backend
    pub async fn create_storage(self) -> Result<Box<dyn SimulationStorage>, StorageError> {
        match self {
            StorageConfig::Memory => {
                Ok(Box::new(MemoryStorage::new()))
            }
            StorageConfig::Redis { url, ttl_seconds } => {
                Ok(Box::new(RedisStorage::new(&url, ttl_seconds).await?))
            }
        }
    }
}
