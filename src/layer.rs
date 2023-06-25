use crate::{neuron::Neuron, value::Value};

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>
}

impl Layer {
    pub fn new(n_in:usize, n_out:usize) -> Layer {
        // let mut neurons: Vec<Neuron> = Vec::new();
        // for _ in 0..n_out {
        //     neurons.push(Neuron::new(n_in));
        // }
        // Layer {
        //     neurons,
        // }
        Layer {
            neurons: (0..n_out)
            .map(|_| Neuron::new(n_in))
            .collect(),
        }
    }

    pub fn forward(&self, xs: &Vec<Value>) -> Vec<Value> {
        //for each neuron, call the forward method in it!
        self.neurons.iter().map(|n| n.forward(xs)).collect()

        // //NON RUST Way
        // let mut f_pass = Vec<Value>::new();
        // for n in self.neurons.clone() {
        //     f_pass.push(n.forward(xs));
        // }
        // f_pass
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
            // //NON RUST Way
            // let mut params = Vec<Value>::new();
            // for n in self.neurons.clone() {
            //     params.push(n.parameters());
            // }
            // params
    }
}