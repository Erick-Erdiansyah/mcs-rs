use candle_core::{Device, Tensor};

mod attention;
mod embedding;
mod helper;
mod math;

use math::{dot, softmax_rows, transpose};

use crate::math::divide_matrix;

fn main() {
    let mut w = 0.5;
    let x = 2.0;
    let target = 10.0;
    let learning_rate = 0.01;

    for epoch in 0..100 {
        let prediction = x * w;
        let loss = (prediction - target) * (prediction - target);
        let gradient = 2.0 * (prediction - target) * x;
        w = w - learning_rate * gradient;

        println!("Epoch {epoch}: w = {:.4}, loss = {:.4}", w, loss);
    }
}

fn attention_scores(queries: &[Vec<f32>], keys: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut scores = Vec::new();

    for q in queries {
        let mut row = Vec::new();

        for k in keys {
            row.push(dot(q, k));
        }

        scores.push(row);
    }

    scores
}
