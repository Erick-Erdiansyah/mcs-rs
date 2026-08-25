use bevy::prelude::*;

use crate::pong::ball::ball_components::Ball;

const BALL: f32 = 32.0;

pub fn spawn_ball(mut commands: Commands, asset_server: Res<AssetServer>) {
    let texture = asset_server.load("ball.png");
    commands.spawn((
        Sprite {
            image: texture,
            ..Default::default()
        },
        Transform {
            translation: Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            ..Default::default()
        },
        Ball {
            velocity: Vec2 { x: 0.0, y: 0.0 },
        },
    ));
}

pub fn move_ball(mut query: Query<(&mut Transform, &mut Ball)>, time: Res<Time>) {
    let delta_time = time.delta_secs();
    for (mut transform, mut ball) in &mut query {
        ball.velocity.y += -150.0 * delta_time;
        transform.translation += ball.velocity.extend(0.0) * delta_time;
    }
}

pub fn update_ball(mut query: Query<(&mut Transform, &mut Ball)>, window: Single<&mut Window>) {
    let height = window.height();
    let width = window.width();
    let half_ball = BALL / 2.0;
    let top = height / 2.0 - half_ball;
    let bottom = -height / 2.0 + half_ball;
    let left = -width / 2.0 - half_ball;
    let right = width / 2.0 + half_ball;
    for (mut transform, mut ball) in &mut query {
        let translation = transform.translation;
        if translation.y <= bottom {
            transform.translation.y = bottom;
            ball.velocity.y *= -1.0;
        }
        if translation.y >= top {
            transform.translation.y = top;
            ball.velocity.y *= -1.0;
        }
        if translation.x <= left {
            transform.translation.x = left;
            ball.velocity.x *= -1.0;
        }
        if translation.x >= right {
            transform.translation.x = right;
            ball.velocity.x *= -1.0;
        }
    }
}
