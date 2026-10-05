use crate::ai::{
    div, exp,
    helper::{Matrix, ValueRef},
    sub,
    torch::Value,
};

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (i, j) in a.iter().zip(b.iter()) {
        sum += i * j
    }
    sum
}

pub fn softmax(scores: &[f32]) -> Vec<f32> {
    // find max score
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

pub fn softmax_value(scores: &Vec<ValueRef>) -> Vec<ValueRef> {
    // find max
    let mut max = f32::NEG_INFINITY;
    for s in scores {
        let current = s.borrow().data;
        if current > max {
            max = current;
        }
    }
    let max_val = Value::new(max);
    let mut exp_scores = Vec::new();
    let mut sum: f32 = 0.0;
    for score in scores {
        let shifted = sub(score, &max_val);
        let e = exp(&shifted);
        exp_scores.push(e.clone());
        sum += e.borrow().data;
    }
    let newsum = Value::new(sum);
    exp_scores.into_iter().map(|x| div(&x, &newsum)).collect()
}

pub fn softmax_rows(matrix: &[Vec<f32>]) -> Matrix {
    let mut result = Vec::new();
    for row in matrix {
        result.push(softmax(row));
    }
    result
}

pub fn matmul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Matrix {
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

pub fn matadd(a: &[Vec<f32>], b: &[Vec<f32>]) -> Matrix {
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

pub fn transpose(matrix: &[Vec<f32>]) -> Matrix {
    let mut result = Vec::new();
    for col in 0..matrix[0].len() {
        let mut row = Vec::new();
        for r in matrix {
            row.push(r[col]);
        }
        result.push(row);
    }
    result
}

pub fn divide_matrix(matrix: &[Vec<f32>], value: f32) -> Matrix {
    let mut result = Vec::new();
    for row in matrix {
        let mut temp = Vec::new();
        for num in row {
            temp.push(num / value);
        }
        result.push(temp);
    }
    result
}
