use candle_core::{Device, Tensor};

mod embedding;
mod math;

use math::{dot, matmul};

fn main() {
    let mut c: Vec<Vec<f32>> = Vec::new();
    c.push(vec![1.0, 2.0]);
    c.push(vec![3.0, 4.0]);
    let mut d: Vec<Vec<f32>> = Vec::new();
    d.push(vec![5.0, 6.0]);
    d.push(vec![7.0, 8.0]);
    println!("{:?}", matmul(&c, &d));
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
