use bevy::app::{Plugin, Startup, Update};

use crate::pong::ball::ball_systems::{move_ball, spawn_ball, update_ball};

pub mod ball_components;
pub mod ball_systems;

pub struct BallPlugin;

impl Plugin for BallPlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_systems(Startup, spawn_ball)
            .add_systems(Update, (move_ball, update_ball));
    }
}
