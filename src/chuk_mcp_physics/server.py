"""Physics MCP Server.

Provides physics calculations and simulations through configurable providers:
- Analytic: Closed-form equations for simple physics problems
- Rapier: Full rigid-body physics simulation using Rapier engine

Features:
- Projectile motion calculations
- Collision detection and prediction
- Force, energy, and momentum calculations
- Full 3D rigid-body simulations with Rapier
- Trajectory recording and R3F-compatible output

All responses use Pydantic models for type safety and validation.
No dictionary goop, no magic strings - everything is strongly typed with enums and constants.
"""

import json
import logging
import sys
from typing import Optional, Union

from chuk_mcp_server import run, tool

from .config import SimulationLimits
from .models import (
    JointDefinition,
    RigidBodyDefinition,
    SimulationConfig,
    SimulationCreateResponse,
    SimulationStepResponse,
    TrajectoryResponse,
    TrajectoryWithEventsResponse,
)
from .providers.factory import get_provider_for_tool
from .analysis import analyze_trajectory_with_events

# Import all tools modules to register their @tool decorated functions
from .tools import (
    basic,
    rotational,
    oscillations,
    circular_motion,
    collisions,
    conservation,
    fluid as fluid_tools,
    kinematics_tools,
    statics,
    convert_units as unit_conversion_tools,
)

# Silence unused import warnings - these imports register @tool decorated functions
_ = (
    basic,
    unit_conversion_tools,
    rotational,
    oscillations,
    circular_motion,
    collisions,
    conservation,
    fluid_tools,
    kinematics_tools,
    statics,
)

# Configure logging
# In STDIO mode, we need to be quiet to avoid polluting the JSON-RPC stream
# Only log to stderr, and only warnings/errors
logging.basicConfig(
    level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s", stream=sys.stderr
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _parse_list(value: Union[list[float], str, None]) -> Optional[list[float]]:
    """Parse list parameter that might come as string from MCP CLI.

    The MCP CLI sometimes sends lists as JSON strings like '[0,1,0]' instead
    of actual Python lists. This helper parses both formats.

    Args:
        value: Either a list, a JSON string, or None

    Returns:
        Parsed list or None
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ============================================================================
# Simulation Tools (Rapier Provider Required)
# ============================================================================


@tool  # type: ignore[arg-type]
async def create_simulation(
    gravity_x: float = 0.0,
    gravity_y: float = -9.81,
    gravity_z: float = 0.0,
    dimensions: int = 3,
    dt: float = 0.016,
    integrator: str = "verlet",
) -> SimulationCreateResponse:
    """Create a new physics simulation using Rapier engine.

    Initializes a new rigid-body physics world with configurable gravity and
    timestep. Returns a simulation ID used for all subsequent operations.

    Args:
        gravity_x: X component of gravity vector (m/s²). Default 0.0
        gravity_y: Y component of gravity vector (m/s²). Default -9.81 (Earth down)
        gravity_z: Z component of gravity vector (m/s²). Default 0.0
        dimensions: 2 or 3 for 2D/3D simulation. Default 3.
        dt: Simulation timestep in seconds. Default 0.016 (60 FPS).
            Smaller = more accurate but slower, larger = faster but less stable
        integrator: Integration method. Options: "euler", "verlet", "rk4". Default "verlet".

    Returns:
        SimulationCreateResponse containing:
            - sim_id: Unique simulation identifier (use for all other sim calls)
            - config: Echo of the configuration used

    Tips for LLMs:
        - Keep simulation IDs in memory for the conversation session
        - Default gravity is Earth standard (9.81 m/s² down = -Y direction)
        - dt=0.016 ≈ 60 FPS, dt=0.008 ≈ 120 FPS (higher accuracy)
        - "verlet" integrator is good default (stable, energy-conserving)
        - Remember to destroy_simulation when done to free resources

    Requires:
        - Rapier provider must be configured (see config.py)
        - Rapier service must be running (see RAPIER_SERVICE.md)

    Example:
        # Create simulation with Earth gravity
        sim = await create_simulation(
            gravity_y=-9.81,
            dt=0.016
        )
        # Use sim.sim_id for add_body, step_simulation, etc.
    """
    # Validate dt
    if dt < SimulationLimits.MIN_DT:
        raise ValueError(
            f"dt={dt} is too small. Minimum is {SimulationLimits.MIN_DT}s. "
            f"Very small timesteps provide diminishing returns and slow performance."
        )
    if dt > SimulationLimits.MAX_DT:
        raise ValueError(
            f"dt={dt} is too large. Maximum is {SimulationLimits.MAX_DT}s. "
            f"Large timesteps cause instability. Recommended: {SimulationLimits.RECOMMENDED_DT}s (60 FPS)."
        )

    config = SimulationConfig(
        gravity=[gravity_x, gravity_y, gravity_z],
        dimensions=dimensions,
        dt=dt,
        integrator=integrator,  # type: ignore
    )
    provider = get_provider_for_tool("create_simulation")
    return await provider.create_simulation(config)


@tool  # type: ignore[arg-type]
async def add_rigid_body(
    sim_id: str,
    body_id: str,
    body_type: str,
    shape: str,
    size: Union[list[float], str, None] = None,
    mass: float = 1.0,
    normal: Union[list[float], str, None] = None,
    offset: Optional[float] = None,
    position: Union[list[float], str, None] = None,
    orientation: Union[list[float], str, None] = None,
    velocity: Union[list[float], str, None] = None,
    angular_velocity: Union[list[float], str, None] = None,
    restitution: float = 0.5,
    friction: float = 0.5,
    is_sensor: bool = False,
    linear_damping: float = 0.0,
    angular_damping: float = 0.0,
    drag_coefficient: Optional[float] = None,
    drag_area: Optional[float] = None,
    drag_axis_ratios: Union[list[float], str, None] = None,
    fluid_density: float = 1.225,
) -> str:
    """Add a rigid body to an existing simulation.

    Creates a new physics body (static, dynamic, or kinematic) with specified
    shape, mass, and initial conditions. Bodies interact via collisions.

    Args:
        sim_id: Simulation ID from create_simulation
        body_id: Unique identifier for this body (user-defined string)
        body_type: "static", "dynamic", or "kinematic"
            - static: Never moves (ground, walls)
            - dynamic: Affected by forces and collisions
            - kinematic: Moves but not affected by forces (scripted motion)
        shape: Collider shape: "box", "sphere", "capsule", "cylinder", "plane"
        size: Shape dimensions:
            - box: [width, height, depth]
            - sphere: [radius]
            - capsule: [half_height, radius]
            - cylinder: [half_height, radius]
            - plane: not needed (use normal/offset instead)
        mass: Mass in kilograms (for dynamic bodies). Default 1.0
        normal: Normal vector [x, y, z] for plane shape. Default [0, 1, 0] (upward)
        offset: Offset along normal for plane. Default 0.0
        position: Initial position [x, y, z]. Default [0, 0, 0]
        orientation: Initial orientation quaternion [x, y, z, w]. Default [0, 0, 0, 1] (identity)
        velocity: Initial linear velocity [x, y, z]. Default [0, 0, 0]
        angular_velocity: Initial angular velocity [x, y, z]. Default [0, 0, 0]
        restitution: Bounciness (0.0 = no bounce, 1.0 = perfect bounce). Default 0.5
        friction: Surface friction (0.0 = ice, 1.0 = rubber). Default 0.5
        is_sensor: If true, detects collisions but doesn't respond physically. Default false
        linear_damping: Linear velocity damping (0.0-1.0) - like air resistance. Default 0.0
        angular_damping: Angular velocity damping (0.0-1.0) - like rotational friction. Default 0.0
        drag_coefficient: Base drag coefficient (Cd) for orientation-dependent drag. Optional
        drag_area: Reference cross-sectional area (m²) for drag calculation. Optional
        drag_axis_ratios: Drag variation along body axes [x, y, z]. E.g., [1.0, 0.2, 1.0] for streamlined along Y. Optional
        fluid_density: Fluid density (kg/m³). Air=1.225, Water=1000. Default 1.225

    Returns:
        body_id (echo of the input ID)

    Tips for LLMs:
        - Create ground FIRST: body_type="static", shape="plane", normal=[0, 1, 0]
        - Box size is full width/height/depth (not half-extents)
        - Sphere size is [radius] (array with one element)
        - Quaternions: identity = [0, 0, 0, 1] (no rotation)
        - Common restitution: steel=0.8, wood=0.5, clay=0.1
        - Common friction: ice=0.05, wood=0.4, rubber=1.0

    Example:
        # Add a ground plane
        await add_rigid_body(
            sim_id=sim_id,
            body_id="ground",
            body_type="static",
            shape="plane",
            normal=[0, 1, 0]
        )

        # Add a bouncing ball
        await add_rigid_body(
            sim_id=sim_id,
            body_id="ball",
            body_type="dynamic",
            shape="sphere",
            size=[0.5],  # radius = 0.5m
            mass=1.0,
            position=[0, 10, 0],
            restitution=0.7
        )

        # Add a falling box
        await add_rigid_body(
            sim_id=sim_id,
            body_id="box",
            body_type="dynamic",
            shape="box",
            size=[1.0, 1.0, 1.0],
            mass=10.0,
            position=[0.0, 5.0, 0.0]
        )
    """
    # Parse list parameters (they might come as JSON strings from MCP CLI)
    parsed_size = _parse_list(size)
    parsed_normal = _parse_list(normal)
    parsed_position = _parse_list(position) or [0.0, 0.0, 0.0]
    parsed_orientation = _parse_list(orientation) or [0.0, 0.0, 0.0, 1.0]
    parsed_velocity = _parse_list(velocity) or [0.0, 0.0, 0.0]
    parsed_angular_velocity = _parse_list(angular_velocity) or [0.0, 0.0, 0.0]
    parsed_drag_axis_ratios = _parse_list(drag_axis_ratios)

    body = RigidBodyDefinition(
        id=body_id,
        kind=body_type,  # type: ignore
        shape=shape,  # type: ignore
        mass=mass,
        size=parsed_size,
        normal=parsed_normal,
        offset=offset,
        position=parsed_position,
        orientation=parsed_orientation,
        velocity=parsed_velocity,
        angular_velocity=parsed_angular_velocity,
        restitution=restitution,
        friction=friction,
        is_sensor=is_sensor,
        linear_damping=linear_damping,
        angular_damping=angular_damping,
        drag_coefficient=drag_coefficient,
        drag_area=drag_area,
        drag_axis_ratios=parsed_drag_axis_ratios,
        fluid_density=fluid_density,
    )
    provider = get_provider_for_tool("add_body")
    return await provider.add_body(sim_id, body)


@tool  # type: ignore[arg-type]
async def add_joint(
    sim_id: str,
    joint: JointDefinition,
) -> str:
    """Add a joint/constraint to connect two rigid bodies.

    Joints allow you to constrain the motion between bodies:
    - FIXED: Rigid connection (glue objects together)
    - REVOLUTE: Hinge rotation around an axis (doors, pendulums)
    - SPHERICAL: Ball-and-socket rotation (ragdolls, gimbals)
    - PRISMATIC: Sliding along an axis (pistons, elevators)

    Args:
        sim_id: Simulation identifier
        joint: Joint definition with type and parameters

    Returns:
        joint_id: Unique identifier for the created joint

    Example - Simple Pendulum:
        # Create fixed anchor point
        add_rigid_body(
            sim_id=sim_id,
            body_id="anchor",
            body_type="static",
            shape="sphere",
            size=[0.05],
            position=[0.0, 5.0, 0.0],
        )

        # Create pendulum bob
        add_rigid_body(
            sim_id=sim_id,
            body_id="bob",
            body_type="dynamic",
            shape="sphere",
            size=[0.1],
            mass=1.0,
            position=[0.0, 3.0, 0.0],
        )

        # Connect with revolute joint (hinge)
        add_joint(
            sim_id=sim_id,
            joint=JointDefinition(
                id="pendulum_joint",
                joint_type="revolute",
                body_a="anchor",
                body_b="bob",
                anchor_a=[0.0, 0.0, 0.0],  # Center of anchor
                anchor_b=[0.0, 0.1, 0.0],   # Top of bob
                axis=[0.0, 0.0, 1.0],        # Rotate around Z-axis
            ),
        )
    """
    provider = get_provider_for_tool("add_joint")
    joint_id = await provider.add_joint(sim_id, joint)
    return joint_id


@tool  # type: ignore[arg-type]
async def step_simulation(
    sim_id: str,
    steps: int = 1,
    dt: Optional[float] = None,
) -> SimulationStepResponse:
    """Step the simulation forward in time.

    Advances the physics simulation by running the integrator for N steps.
    Returns the complete state of all bodies after stepping.

    Args:
        sim_id: Simulation ID
        steps: Number of timesteps to simulate. Default 1.
            Example: steps=600 with dt=0.016 = 9.6 seconds of simulation
        dt: Optional timestep override (seconds). If None, uses config default.

    Returns:
        SimulationStepResponse containing:
            - sim_id: Simulation identifier
            - time: Current simulation time in seconds
            - bodies: List of all body states with positions, velocities, contacts

    Tips for LLMs:
        - Each body state includes position, orientation (quaternion), velocities
        - contacts array shows active collisions with impulse magnitudes
        - For real-time preview: steps=1, call repeatedly
        - For final result: steps=1000+, call once
        - Large step counts may timeout - limit to ~10,000 steps per call

    Example:
        # Simulate 10 seconds at 60 FPS
        result = await step_simulation(
            sim_id=sim_id,
            steps=600  # 600 steps × 0.016s = 9.6s
        )
        for body in result.bodies:
            print(f"{body.id}: position={body.position}")
    """
    # Validate steps
    if steps > SimulationLimits.MAX_STEPS_PER_CALL:
        raise ValueError(
            f"Requested {steps} steps exceeds maximum of {SimulationLimits.MAX_STEPS_PER_CALL}. "
            f"Break into multiple calls or reduce steps to avoid timeout."
        )
    if steps < 1:
        raise ValueError("steps must be at least 1")

    # Validate dt if provided
    if dt is not None:
        if dt < SimulationLimits.MIN_DT:
            raise ValueError(f"dt={dt} is too small. Minimum is {SimulationLimits.MIN_DT}s")
        if dt > SimulationLimits.MAX_DT:
            raise ValueError(f"dt={dt} is too large. Maximum is {SimulationLimits.MAX_DT}s")

    provider = get_provider_for_tool("step_simulation")
    return await provider.step_simulation(sim_id, steps, dt)


@tool  # type: ignore[arg-type]
async def record_trajectory(
    sim_id: str,
    body_id: str,
    steps: int,
    dt: Optional[float] = None,
) -> TrajectoryResponse:
    """Record the trajectory of a specific body over time.

    Steps the simulation and records position/orientation/velocity at each
    timestep for one body. Perfect for generating animation data for R3F.

    Args:
        sim_id: Simulation ID
        body_id: ID of the body to track
        steps: Number of timesteps to record
        dt: Optional timestep override. If None, uses config default.

    Returns:
        TrajectoryResponse containing:
            - body_id: Tracked body identifier
            - frames: List of trajectory frames with time, position, orientation, velocity
            - total_time: Total simulated time in seconds
            - num_frames: Number of frames recorded

    Tips for LLMs:
        - Each frame has: time, position [x,y,z], orientation [x,y,z,w], velocity [x,y,z]
        - Frames are evenly spaced in time (every dt seconds)
        - Output is R3F-compatible: use position/orientation directly in Three.js
        - For 60 FPS video: record at dt=1/60 ≈ 0.0167
        - Typical recording: 100-1000 frames (1.6-16 seconds at 60 FPS)

    Example:
        # Record 5 seconds of a falling ball
        traj = await record_trajectory(
            sim_id=sim_id,
            body_id="ball",
            steps=300  # 300 × 0.016 ≈ 5 seconds
        )
        # Use traj.frames in React Three Fiber for animation
    """
    # Validate steps/frames
    if steps > SimulationLimits.MAX_TRAJECTORY_FRAMES:
        raise ValueError(
            f"Requested {steps} frames exceeds maximum of {SimulationLimits.MAX_TRAJECTORY_FRAMES}. "
            f"Reduce frame count to avoid excessive memory usage."
        )
    if steps < 1:
        raise ValueError("steps must be at least 1")

    # Validate dt if provided
    if dt is not None:
        if dt < SimulationLimits.MIN_DT:
            raise ValueError(f"dt={dt} is too small. Minimum is {SimulationLimits.MIN_DT}s")
        if dt > SimulationLimits.MAX_DT:
            raise ValueError(f"dt={dt} is too large. Maximum is {SimulationLimits.MAX_DT}s")

    provider = get_provider_for_tool("record_trajectory")
    return await provider.record_trajectory(sim_id, body_id, steps, dt)


@tool  # type: ignore[arg-type]
async def record_trajectory_with_events(
    sim_id: str,
    body_id: str,
    steps: int,
    dt: Optional[float] = None,
    detect_bounces: bool = True,
    bounce_height_threshold: float = 0.01,
) -> TrajectoryWithEventsResponse:
    """Record trajectory and automatically detect collision and bounce events.

    This is an enhanced version of record_trajectory that analyzes the motion
    and detects important events like bounces and collisions. Perfect for
    answering questions like "how many times did the ball bounce?"

    Args:
        sim_id: Simulation ID
        body_id: Body to track
        steps: Number of simulation steps to record
        dt: Optional custom timestep (overrides simulation default)
        detect_bounces: Whether to detect bounce events (default True)
        bounce_height_threshold: Maximum height to consider as "on ground" in meters (default 0.01)

    Returns:
        TrajectoryWithEventsResponse containing:
            - frames: Trajectory frames (positions, velocities)
            - bounces: Detected bounce events with energy loss
            - contact_events: Contact/collision events (future)

    Tips for LLMs:
        - Use this instead of record_trajectory when you need event detection
        - Bounces are detected from velocity reversals near the ground
        - Each bounce includes: time, position, speeds before/after, energy loss
        - Use `trajectory.bounces` to count or analyze bounces
        - Adjust bounce_height_threshold for different ground shapes

    Example:
        # Record ball bouncing and count bounces
        traj = await record_trajectory_with_events(
            sim_id=sim_id,
            body_id="ball",
            steps=600,
            detect_bounces=True,
            bounce_height_threshold=0.01  # 1cm threshold
        )
        print(f"Detected {len(traj.bounces)} bounces")
        for bounce in traj.bounces:
            print(f"Bounce #{bounce.bounce_number} at t={bounce.time:.2f}s")
    """
    # Validate steps/frames
    if steps > SimulationLimits.MAX_TRAJECTORY_FRAMES:
        raise ValueError(
            f"Requested {steps} frames exceeds maximum of {SimulationLimits.MAX_TRAJECTORY_FRAMES}. "
            f"Reduce frame count to avoid excessive memory usage."
        )
    if steps < 1:
        raise ValueError("steps must be at least 1")

    # Validate dt if provided
    if dt is not None:
        if dt < SimulationLimits.MIN_DT:
            raise ValueError(f"dt={dt} is too small. Minimum is {SimulationLimits.MIN_DT}s")
        if dt > SimulationLimits.MAX_DT:
            raise ValueError(f"dt={dt} is too large. Maximum is {SimulationLimits.MAX_DT}s")

    # Get regular trajectory
    provider = get_provider_for_tool("record_trajectory")
    trajectory = await provider.record_trajectory(sim_id, body_id, steps, dt)

    # Analyze and detect events
    return analyze_trajectory_with_events(
        frames=trajectory.frames,
        dt=trajectory.dt,
        body_id=trajectory.meta.body_id,
        total_time=trajectory.meta.total_time,
        detect_bounces_enabled=detect_bounces,
        bounce_height_threshold=bounce_height_threshold,
        contact_events=None,  # Future: get from Rapier service
    )


@tool  # type: ignore[arg-type]
async def destroy_simulation(sim_id: str) -> str:
    """Destroy a simulation and free resources.

    Cleanup when done with a simulation. Important for long-running servers
    to avoid memory leaks.

    Args:
        sim_id: Simulation ID to destroy

    Returns:
        Success message

    Tips for LLMs:
        - Always destroy simulations when conversation ends or changes topic
        - Rapier service keeps simulations in memory until explicitly destroyed
        - Good practice: destroy after recording trajectory or final state

    Example:
        await destroy_simulation(sim_id)
    """
    provider = get_provider_for_tool("destroy_simulation")
    await provider.destroy_simulation(sim_id)
    return f"Simulation {sim_id} destroyed"


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:  # pragma: no cover
    """Run the Physics MCP server.

    Supports two transport modes:
    - stdio: For MCP clients (default) - Claude Desktop, mcp-cli
    - http: For web/REST access (use --http or http argument)
    """
    # Check if transport is specified in command line args
    # Default to stdio for MCP compatibility (Claude Desktop, mcp-cli)
    transport = "stdio"

    # Allow HTTP mode via command line
    if len(sys.argv) > 1 and sys.argv[1] in ["http", "--http"]:
        transport = "http"
        # Only log in HTTP mode
        logger.warning("Starting Physics MCP Server in HTTP mode")

    # Suppress chuk_mcp_server logging in STDIO mode
    if transport == "stdio":
        # Set chuk_mcp_server loggers to ERROR only
        logging.getLogger("chuk_mcp_server").setLevel(logging.ERROR)
        logging.getLogger("chuk_mcp_server.core").setLevel(logging.ERROR)
        logging.getLogger("chuk_mcp_server.stdio_transport").setLevel(logging.ERROR)
        # Suppress httpx logging (API calls to Rapier service)
        logging.getLogger("httpx").setLevel(logging.ERROR)

    # For HTTP mode, bind to 0.0.0.0 for container/cloud deployment
    # This is intentional for Docker/Fly.io deployment and is safe when:
    # 1. Running behind a reverse proxy (nginx, Fly.io edge)
    # 2. Network-level firewall rules are in place
    # 3. Not exposing directly to the internet
    if transport == "http":
        import os

        run(transport=transport, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))  # nosec B104
    else:
        run(transport=transport)


if __name__ == "__main__":  # pragma: no cover
    main()
