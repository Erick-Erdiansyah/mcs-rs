use std::{cell::RefCell, rc::Rc};

use candle_core::{D, Device, Tensor};

mod attention;
mod embedding;
mod helper;
mod math;
mod neuron;
mod torch;

use math::{dot, softmax_rows, transpose};
use neuron::Neuron;

use crate::{
    helper::ValueRef,
    math::divide_matrix,
    neuron::MLP,
    torch::{Op, Value},
};

fn main() {
    let model = MLP::new(3, &[3, 1]);

    let inputs = vec![Value::new(2.0), Value::new(3.0), Value::new(4.0)];

    let target = Value::new(1.0);

    let weight = &model.layers[0].neurons[1].weight[2];

    let original = weight.borrow().data;
    let epsilon = 0.0001;

    // -------------------------
    // Numerical gradient
    // -------------------------

    weight.borrow_mut().data = original + epsilon;

    let output = model.forward(&inputs);
    let prediction = &output[0];
    let error = sub(prediction, &target);
    let loss_plus = pow(&error, 2.0);
    let loss_plus = loss_plus.borrow().data;

    weight.borrow_mut().data = original;

    let output = model.forward(&inputs);
    let prediction = &output[0];
    let error = sub(prediction, &target);
    let loss = pow(&error, 2.0);

    let loss_value = loss.borrow().data;

    // -------------------------
    // Analytical gradient
    // -------------------------

    model.zero_grad();

    backward(loss);

    let analytical_grad = weight.borrow().grad;

    // -------------------------
    // Compare
    // -------------------------

    let numerical_grad = (loss_plus - loss_value) / epsilon;

    println!("numerical:  {}", numerical_grad);
    println!("analytical: {}", analytical_grad);
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

pub fn mul(a: &ValueRef, b: &ValueRef) -> ValueRef {
    Rc::new(RefCell::new(Value {
        data: a.borrow().data * b.borrow().data,
        grad: 0.0,
        prev: vec![Rc::clone(a), Rc::clone(b)],
        op: torch::Op::Mul,
        visited: false,
        extra: None,
    }))
}

pub fn add(a: &ValueRef, b: &ValueRef) -> ValueRef {
    Rc::new(RefCell::new(Value {
        data: a.borrow().data + b.borrow().data,
        grad: 0.0,
        prev: vec![Rc::clone(a), Rc::clone(b)],
        op: torch::Op::Add,
        visited: false,
        extra: None,
    }))
}

pub fn sub(a: &ValueRef, b: &ValueRef) -> ValueRef {
    Rc::new(RefCell::new(Value {
        data: a.borrow().data - b.borrow().data,
        grad: 0.0,
        prev: vec![Rc::clone(a), Rc::clone(b)],
        op: torch::Op::Sub,
        visited: false,
        extra: None,
    }))
}

pub fn pow(a: &ValueRef, b: f32) -> ValueRef {
    let data = a.borrow().data.powf(b);
    Rc::new(RefCell::new(Value {
        data,
        grad: 0.0,
        prev: vec![Rc::clone(a)],
        op: torch::Op::Pow,
        visited: false,
        extra: Some(b),
    }))
}

pub fn tanh(a: &ValueRef) -> ValueRef {
    let data = a.borrow().data.tanh();
    Rc::new(RefCell::new(Value {
        data,
        grad: 0.0,
        prev: vec![Rc::clone(a)],
        op: torch::Op::Tanh,
        visited: false,
        extra: None,
    }))
}

pub fn topo_ordering(root: ValueRef, order: &mut Vec<ValueRef>) {
    let visited = root.borrow().visited;
    let parents = root.borrow().prev.clone();
    if visited == true {
        return;
    }
    for parent in parents {
        topo_ordering(parent, order);
    }
    root.borrow_mut().visited = true;
    order.push(Rc::clone(&root));
}

pub fn set_grad(order: &mut Vec<ValueRef>) {
    for value in order {
        value.borrow_mut().grad = 1.0;
    }
}

pub fn backward(root: ValueRef) {
    let mut order = Vec::new();
    root.borrow_mut().grad = 1.0;
    topo_ordering(root, &mut order);
    order.reverse();
    for value in order {
        let (op, parents, current_grad, output, extra) = {
            let node = value.borrow();
            (node.op, node.prev.clone(), node.grad, node.data, node.extra)
        };
        match op {
            Op::Mul => {
                let mut left = parents[0].borrow_mut();
                let mut right = parents[1].borrow_mut();

                left.grad += current_grad * right.data;
                right.grad += current_grad * left.data;
            }
            Op::Add => {
                parents[0].borrow_mut().grad += current_grad;
                parents[1].borrow_mut().grad += current_grad;
            }
            Op::Sub => {
                parents[0].borrow_mut().grad += current_grad;
                parents[1].borrow_mut().grad -= current_grad;
            }
            Op::Div => {
                let mut left = parents[0].borrow_mut();
                let mut right = parents[1].borrow_mut();
                left.grad += current_grad * (1.0 / right.data);
                right.grad += current_grad * (-left.data / (right.data * right.data));
            }
            Op::Pow => {
                let mut parent = parents[0].borrow_mut();
                match extra {
                    Some(exponent) => {
                        parent.grad += current_grad * exponent * parent.data.powf(exponent - 1.0);
                    }
                    None => println!("something is wrong... I think"),
                }
            }
            Op::ReLU => {
                let mut parent = parents[0].borrow_mut();
                if parent.data > 0.0 {
                    parent.grad += current_grad;
                }
            }
            Op::Tanh => {
                parents[0].borrow_mut().grad += current_grad * (1.0 - output * output);
            }
            Op::Exp => {
                parents[0].borrow_mut().grad += current_grad * output;
            }
            Op::None => {}
        }
    }
}