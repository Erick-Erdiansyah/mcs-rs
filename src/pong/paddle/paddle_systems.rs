use bevy::prelude::*;

use crate::pong::paddle::paddle_components::Paddle;

pub fn spawn_paddle(mut commands: Commands, asset_server: Res<AssetServer>) {
    let texture = asset_server.load("pad.png");
    commands.spawn((
        Sprite {
            image: texture,
            ..Default::default()
        },
        Transform {
            translation: Vec3 {
                x: -50.0,
                y: -250.0,
                z: 0.0,
            },
            ..Default::default()
        },
        Paddle,
    ));
}

pub fn move_paddle(
    mut query: Query<(&mut Transform, &Paddle)>,
    keyboard_input: Res<ButtonInput<KeyCode>>,
    window: Single<&mut Window>,
    time: Res<Time>,
) {
    let width = window.width();
    let left = -width / 2.0;
    let right = width / 2.0;

    for (mut transform, _) in &mut query {
        if keyboard_input.pressed(KeyCode::KeyA) {
            transform.translation.x -= 200.0 * time.delta_secs();
            if transform.translation.x <= left {
                transform.translation.x = left;
            }
        }
        if keyboard_input.pressed(KeyCode::KeyD) {
            transform.translation.x += 200.0 * time.delta_secs();
            if transform.translation.x >= right {
                transform.translation.x = right;
            }
        }
    }
}
