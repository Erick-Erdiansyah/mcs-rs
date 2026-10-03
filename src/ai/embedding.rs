pub struct Embedding {
    pub weights: Vec<Vec<f32>>,
}

impl Embedding {
    pub fn forward(&self, token: &[usize]) -> Vec<Vec<f32>> {
        let mut result = vec![];
        for t in token {
            result.push(self.weights[*t].clone());
        }
        result
    }
}
