use bevy::app::{Plugin, Startup, Update};

use crate::pong::paddle::paddle_systems::{move_paddle, spawn_paddle};

pub mod paddle_components;
pub mod paddle_systems;

pub struct PaddlePlugin;

impl Plugin for PaddlePlugin {
    fn build(&self, app: &mut bevy::app::App) {
        app.add_systems(Startup, spawn_paddle)
            .add_systems(Update, move_paddle);
    }
}
