pub struct Embedding {
    pub weights: Vec<Vec<f32>>,
}

impl Embedding {
    pub fn forward(&self, token: &[usize]) -> Vec<Vec<f32>> {
        let mut result = vec![];
        for test in token {
            result.push(self.weights[*test].clone());
        }
        result
    }
}
