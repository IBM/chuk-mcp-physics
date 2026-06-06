use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{info, warn};
use uuid::Uuid;

mod physics;
mod storage;

use physics::{Simulation, SimulationConfig};
use storage::{SimulationStorage, StorageConfig};

// Application state - now uses the storage abstraction
type AppState = Arc<RwLock<Box<dyn SimulationStorage>>>;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rapier_physics_service=info,tower_http=info".into()),
        )
        .init();

    // Initialize storage backend from environment
    let storage_config = StorageConfig::from_env();
    let storage = storage_config.create_storage()
        .await
        .expect("Failed to initialize storage backend");

    let simulations: AppState = Arc::new(RwLock::new(storage));

    // Build router
    let app = Router::new()
        .route("/health", get(health_check))
        .route("/simulations", post(create_simulation))
        .route("/simulations/:sim_id/state", get(get_simulation_state))
        .route("/simulations/:sim_id", delete(destroy_simulation))
        .route("/simulations/:sim_id/bodies", post(add_body))
        .route("/simulations/:sim_id/joints", post(add_joint))
        .route("/simulations/:sim_id/step", post(step_simulation))
        .route("/simulations/:sim_id/bodies/:body_id/trajectory", post(record_trajectory))
        .layer(CorsLayer::permissive())
        .with_state(simulations);

    // Start server (PORT is assigned by the host fleet; default 9000 standalone)
    let port = std::env::var("PORT").unwrap_or_else(|_| "9000".to_string());
    let addr = format!("0.0.0.0:{}", port);
    info!("🦀 Rapier Physics Service starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr.as_str()).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "rapier-physics-service",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

// Request/Response types
#[derive(Debug, Deserialize)]
struct CreateSimulationRequest {
    gravity: [f32; 3],
    dimensions: u8,
    #[serde(default = "default_dt")]
    dt: f32,
    #[serde(default)]
    integrator: String,
}

fn default_dt() -> f32 {
    0.016
}

#[derive(Debug, Serialize)]
struct CreateSimulationResponse {
    sim_id: String,
    config: SimulationConfigResponse,
}

#[derive(Debug, Serialize)]
struct SimulationConfigResponse {
    gravity: [f32; 3],
    dimensions: u8,
    dt: f32,
    integrator: String,
}

#[derive(Debug, Deserialize)]
struct AddBodyRequest {
    id: String,
    kind: String, // "static", "dynamic", "kinematic"
    shape: String, // "box", "sphere", "capsule", "plane"
    #[serde(default)]
    size: Vec<f32>,
    #[serde(default)]
    mass: Option<f32>,
    #[serde(default)]
    position: Option<[f32; 3]>,
    #[serde(default)]
    orientation: Option<[f32; 4]>, // quaternion [x, y, z, w]
    #[serde(default)]
    velocity: Option<[f32; 3]>,
    #[serde(default)]
    angular_velocity: Option<[f32; 3]>,
    #[serde(default = "default_friction")]
    friction: f32,
    #[serde(default = "default_restitution")]
    restitution: f32,
    #[serde(default)]
    normal: Option<[f32; 3]>, // For plane: normal vector
    #[serde(default)]
    offset: Option<f32>, // For plane: distance from origin
    #[serde(default)]
    linear_damping: Option<f32>, // Linear velocity damping (Phase 1.4)
    #[serde(default)]
    angular_damping: Option<f32>, // Angular velocity damping (Phase 1.4)
    // Orientation-dependent drag parameters (Phase 2)
    #[serde(default)]
    drag_coefficient: Option<f32>, // Base drag coefficient (Cd)
    #[serde(default)]
    drag_area: Option<f32>, // Reference cross-sectional area (m²)
    #[serde(default)]
    drag_axis_ratios: Option<[f32; 3]>, // Drag variation along body axes [x, y, z]
    #[serde(default)]
    fluid_density: Option<f32>, // Fluid density (kg/m³), default 1.225 for air
}

fn default_friction() -> f32 {
    0.5
}

fn default_restitution() -> f32 {
    0.3
}

#[derive(Debug, Serialize)]
struct AddBodyResponse {
    body_id: String,
}

#[derive(Debug, Deserialize)]
struct StepSimulationRequest {
    steps: usize,
    #[serde(default)]
    dt: Option<f32>,
}

#[derive(Debug, Serialize)]
struct SimulationStateResponse {
    sim_id: String,
    time: f32,
    bodies: Vec<BodyStateResponse>,
}

#[derive(Debug, Serialize)]
struct BodyStateResponse {
    id: String,
    position: [f32; 3],
    orientation: [f32; 4],
    velocity: [f32; 3],
    angular_velocity: [f32; 3],
    contacts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RecordTrajectoryRequest {
    steps: usize,
    #[serde(default)]
    dt: Option<f32>,
}

#[derive(Debug, Serialize)]
struct TrajectoryResponse {
    body_id: String,
    frames: Vec<TrajectoryFrame>,
    total_time: f32,
    num_frames: usize,
    #[serde(default)]
    contact_events: Vec<physics::ContactEventData>,
}

#[derive(Debug, Clone, Serialize)]
struct TrajectoryFrame {
    time: f32,
    position: [f32; 3],
    orientation: [f32; 4],
    velocity: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct AddJointRequest {
    id: String,
    joint_type: String,
    body_a: String,
    body_b: String,
    #[serde(default = "default_anchor")]
    anchor_a: [f32; 3],
    #[serde(default = "default_anchor")]
    anchor_b: [f32; 3],
    #[serde(default)]
    axis: Option<[f32; 3]>,
    #[serde(default)]
    limits: Option<[f32; 2]>,
}

fn default_anchor() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

#[derive(Debug, Serialize)]
struct AddJointResponse {
    joint_id: String,
}

// Handlers
async fn create_simulation(
    State(state): State<AppState>,
    Json(req): Json<CreateSimulationRequest>,
) -> Result<Json<CreateSimulationResponse>, StatusCode> {
    let sim_id = Uuid::new_v4().to_string();

    let config = SimulationConfig {
        gravity: req.gravity,
        dimensions: req.dimensions,
        dt: req.dt,
        integrator: if req.integrator.is_empty() {
            "verlet".to_string()
        } else {
            req.integrator
        },
    };

    let simulation = Simulation::new(config.clone());

    info!("Created simulation {}", sim_id);
    state.write().await.upsert(&sim_id, simulation).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(CreateSimulationResponse {
        sim_id: sim_id.clone(),
        config: SimulationConfigResponse {
            gravity: config.gravity,
            dimensions: config.dimensions,
            dt: config.dt,
            integrator: config.integrator,
        },
    }))
}

async fn add_body(
    State(state): State<AppState>,
    Path(sim_id): Path<String>,
    Json(req): Json<AddBodyRequest>,
) -> Result<Json<AddBodyResponse>, StatusCode> {
    let mut storage = state.write().await;
    let sim = storage.get_mut(&sim_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    sim.add_body(
        req.id.clone(),
        req.kind,
        req.shape,
        req.size,
        req.mass,
        req.position,
        req.orientation,
        req.velocity,
        req.angular_velocity,
        req.friction,
        req.restitution,
        req.normal,
        req.offset,
        req.linear_damping,
        req.angular_damping,
        req.drag_coefficient,
        req.drag_area,
        req.drag_axis_ratios,
        req.fluid_density,
    );

    info!("Added body {} to simulation {}", req.id, sim_id);

    Ok(Json(AddBodyResponse { body_id: req.id }))
}

async fn add_joint(
    State(state): State<AppState>,
    Path(sim_id): Path<String>,
    Json(req): Json<AddJointRequest>,
) -> Result<Json<AddJointResponse>, StatusCode> {
    let mut storage = state.write().await;
    let sim = storage.get_mut(&sim_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let joint_def = physics::JointDefinition {
        id: req.id.clone(),
        joint_type: req.joint_type,
        body_a: req.body_a,
        body_b: req.body_b,
        anchor_a: req.anchor_a,
        anchor_b: req.anchor_b,
        axis: req.axis,
        limits: req.limits,
    };

    sim.add_joint(joint_def)
        .map_err(|e| {
            warn!("Failed to add joint: {}", e);
            StatusCode::BAD_REQUEST
        })?;

    info!("Added joint {} to simulation {}", req.id, sim_id);

    Ok(Json(AddJointResponse { joint_id: req.id }))
}

async fn step_simulation(
    State(state): State<AppState>,
    Path(sim_id): Path<String>,
    Json(req): Json<StepSimulationRequest>,
) -> Result<Json<SimulationStateResponse>, StatusCode> {
    let mut storage = state.write().await;
    let sim = storage.get_mut(&sim_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    sim.step(req.steps, req.dt);

    Ok(Json(simulation_state_to_response(sim_id, sim)))
}

async fn get_simulation_state(
    State(state): State<AppState>,
    Path(sim_id): Path<String>,
) -> Result<Json<SimulationStateResponse>, StatusCode> {
    let mut storage = state.write().await;
    let sim = storage.get_mut(&sim_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(simulation_state_to_response(sim_id, sim)))
}

async fn record_trajectory(
    State(state): State<AppState>,
    Path((sim_id, body_id)): Path<(String, String)>,
    Json(req): Json<RecordTrajectoryRequest>,
) -> Result<Json<TrajectoryResponse>, StatusCode> {
    let mut storage = state.write().await;
    let sim = storage.get_mut(&sim_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::NOT_FOUND)?;

    let mut frames = Vec::new();
    let mut all_contact_events = Vec::new();
    let start_time = sim.time;

    for _ in 0..req.steps {
        // Get body state before step
        if let Some(body_state) = sim.get_body_state(&body_id) {
            frames.push(TrajectoryFrame {
                time: sim.time,
                position: body_state.position,
                orientation: body_state.orientation,
                velocity: body_state.velocity,
            });
        }

        // Step simulation (also detects contact events)
        sim.step(1, req.dt);

        // Collect contact events that occurred during this step
        let events = sim.get_and_clear_contact_events();
        all_contact_events.extend(events);
    }

    let total_time = sim.time - start_time;

    Ok(Json(TrajectoryResponse {
        body_id,
        frames: frames.clone(),
        total_time,
        num_frames: frames.len(),
        contact_events: all_contact_events,
    }))
}

async fn destroy_simulation(
    State(state): State<AppState>,
    Path(sim_id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    let mut storage = state.write().await;

    if storage.remove(&sim_id).await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)? {
        info!("Destroyed simulation {}", sim_id);
        Ok(StatusCode::NO_CONTENT)
    } else {
        warn!("Attempted to destroy non-existent simulation {}", sim_id);
        Err(StatusCode::NOT_FOUND)
    }
}

// Helper functions
fn simulation_state_to_response(sim_id: String, sim: &Simulation) -> SimulationStateResponse {
    let bodies = sim
        .get_all_bodies()
        .iter()
        .map(|(id, state)| BodyStateResponse {
            id: id.clone(),
            position: state.position,
            orientation: state.orientation,
            velocity: state.velocity,
            angular_velocity: state.angular_velocity,
            contacts: state.contacts.clone(),
        })
        .collect();

    SimulationStateResponse {
        sim_id,
        time: sim.time,
        bodies,
    }
}
