use std::{cell::RefCell, rc::Rc};

use crate::torch::Value;

pub type Matrix = Vec<Vec<f32>>;
pub type ValueRef = Rc<RefCell<Value>>;