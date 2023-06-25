use crate::{layer::Layer, value::Value};

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(n_in:usize, n_out:Vec<usize>) -> MLP {
        let o_len:usize = n_out.len();
        let sz:Vec<usize> = [n_in].into_iter().chain(n_out).collect();

        MLP {
            layers: (0..o_len)
            .map(|i| Layer::new(sz[i], sz[i+1])).collect(),
        }
    }

    pub fn forward(&self, mut xs: Vec<Value>) -> Vec<Value> {
        for layer in &self.layers {
            xs = layer.forward(&xs);
        }
        xs
    }
    

    pub fn parameters(&self) -> Vec<Value> {
        //use flat map to flatten in a dimension, then map
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }
}
