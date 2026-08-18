use crate::{add, helper::ValueRef, mul, tanh, torch::Value};

pub struct Neuron {
    pub weight: Vec<ValueRef>,
    pub bias: ValueRef,
}

impl Neuron {
    pub fn new(input_size: usize) -> Self {
        Self {
            weight: vec![Value::new(0.5), Value::new(-0.2), Value::new(0.1)],
            bias: Value::new(0.0),
        }
    }
    pub fn forward(&self, input: &[ValueRef]) -> ValueRef {
        let mut result = mul(&self.weight[0], &input[0]);
        for i in 1..input.len() {
            let product = mul(&self.weight[i], &input[i]);
            result = add(&result, &product);
        }
        result = add(&result, &self.bias);
        result = tanh(&result);
        result
    }
    pub fn zero_grad(&self) {
        for weight in &self.weight {
            weight.borrow_mut().grad = 0.0
        }
        self.bias.borrow_mut().grad = 0.0
    }
    pub fn update(&self, learning_rate: f32) {
        for weight in &self.weight {
            let grad = weight.borrow().grad;
            weight.borrow_mut().data -= learning_rate * grad
        }
        let grad = self.bias.borrow().grad;
        self.bias.borrow_mut().data -= learning_rate * grad
    }
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..output_size {
            neurons.push(Neuron::new(input_size));
        }
        Self { neurons }
    }
    pub fn forward(&self, inputs: &[ValueRef]) -> Vec<ValueRef> {
        let mut result = Vec::new();
        for neuron in &self.neurons {
            result.push(neuron.forward(inputs));
        }
        result
    }
    pub fn zero_grad(&self) {
        for neuron in &self.neurons {
            neuron.zero_grad();
        }
    }
    pub fn update(&self, learning_rate: f32) {
        for neuron in &self.neurons {
            neuron.update(learning_rate);
        }
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

// pony-powered neural network
impl MLP {
    pub fn new(input: usize, neuron: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut x = input;
        for i in 0..neuron.len() {
            let l = Layer::new(x, neuron[i]);
            layers.push(l);
            x = neuron[i];
        }
        Self { layers }
    }
    pub fn forward(&self, inputs: &[ValueRef]) -> Vec<ValueRef> {
        let mut current = inputs.to_vec();
        for layer in &self.layers {
            current = layer.forward(&current);
        }
        current
    }
    pub fn zero_grad(&self) {
        for layer in &self.layers {
            layer.zero_grad();
        }
    }
    pub fn update(&self, learning_rate: f32) {
        for layer in &self.layers {
            layer.update(learning_rate);
        }
    }
}
