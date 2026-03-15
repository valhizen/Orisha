/// Handles loading and unloading terrain chunks around the player or camera.
pub mod chunk_manager;
/// Loads mesh data from GLB files into CPU-side vertex/index arrays.
pub mod model_loader;
/// Contains simple terrain collision and movement physics helpers.
pub mod physics;
/// Defines the player state and movement logic.
pub mod player;
/// Builds sky geometry and computes sun direction for the day/night cycle.
pub mod sky;
/// Tracks frame delta time and fixed-step update timing.
pub mod time;
/// Generates procedural terrain heights, colors, water, and chunk meshes.
pub mod world_generation;
