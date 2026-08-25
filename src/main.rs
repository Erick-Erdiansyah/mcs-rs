use bevy::prelude::*;

use crate::pong::{ball::BallPlugin, paddle::PaddlePlugin};

pub mod pong;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, camera)
        .add_plugins(PaddlePlugin)
        .run();
}

pub fn camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}
