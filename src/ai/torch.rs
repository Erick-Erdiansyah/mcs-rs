use std::{cell::RefCell, rc::Rc};

use crate::helper::ValueRef;

#[derive(Debug, Clone, Copy)]
pub enum Op {
    None,
    Add,
    Mul,
    Sub,
    Div,
    Pow,
    ReLU,
    Tanh,
    Exp,
}

#[derive(Debug, Clone)]
pub struct Value {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<ValueRef>,
    pub op: Op,
    pub visited: bool,
    pub extra: Option<f32>,
}

impl Value {
    pub fn new(data: f32) -> ValueRef {
        Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            prev: Vec::new(),
            op: Op::None,
            visited: false,
            extra: None,
        }))
    }
}
