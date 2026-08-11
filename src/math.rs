pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (i, j) in a.iter().zip(b.iter()) {
        sum += i * j
    }
    sum
}

pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_scores = Vec::new();
    let mut sum: f32 = 0.0;
    for score in scores {
        let e = (score - max).exp();
        exp_scores.push(e);
        sum += e;
    }
    exp_scores.into_iter().map(|x| x / sum).collect()
}

pub fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let mut result = Vec::new();
    assert_eq!(a[0].len(), b.len());
    for row in a {
        let mut output_row = Vec::new();
        for (index, _) in b.iter().enumerate() {
            let mut column = Vec::new();
            for c in b {
                column.push(c[index]);
            }
            output_row.push(dot(row, &column));
        }
        result.push(output_row);
    }
    result
}
