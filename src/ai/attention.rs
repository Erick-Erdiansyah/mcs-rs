use rand::RngExt;

use crate::{
    helper::Matrix,
    math::{divide_matrix, matmul, softmax_rows, transpose},
};

pub struct AttentionHead {
    pub wq: Matrix,
    pub wk: Matrix,
    pub wv: Matrix,
    pub embed_dim: usize,
}

pub struct MultiHeadAttention {
    pub heads: Vec<AttentionHead>,
}

impl AttentionHead {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            wq: random_matrix(embed_dim, embed_dim),
            wk: random_matrix(embed_dim, embed_dim),
            wv: random_matrix(embed_dim, embed_dim),
            embed_dim,
        }
    }
    pub fn forward(&self, x: &[Vec<f32>]) -> Matrix {
        let q = matmul(x, &self.wq);
        let k = matmul(x, &self.wk);
        let v = matmul(x, &self.wv);
        let kt = transpose(&k);
        let scores = matmul(&q, &kt);
        let scores = divide_matrix(&scores, (self.embed_dim as f32).sqrt());
        let weight = softmax_rows(&scores);
        let output = matmul(&weight, &v);
        output
    }
}

fn random_matrix(rows: usize, cols: usize) -> Matrix {
    let mut rng = rand::rng();
    let mut matrix = Vec::new();
    for _ in 0..rows {
        let mut row = Vec::new();
        for _ in 0..cols {
            row.push(rng.random_range(-0.1..0.1));
        }
        matrix.push(row);
    }
    matrix
}
