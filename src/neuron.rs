use crate::value::Value;

use rand::{thread_rng, Rng};

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    //neuron only takes in number of inputs, w and b are randomly generated
    pub fn new(n_in: usize) -> Neuron {
        //create a Neuron that has a list of randomly initialized values, and a randomly initialized bias
        let mut rng = thread_rng();
        let mut rand_gen = || {
           let range:f64 = rng.gen_range(-1.0..1.0);
            Value::from(range)
        };

        let mut weights: Vec<Value> = Vec::new();
        for _ in 0..n_in {
            weights.push(rand_gen());
        }

        Neuron {
            weights, 
            bias: rand_gen().label("b"),
        }
    }
    //x is an input layer
    pub fn forward(&self, xs: &Vec<Value>) -> Value {
        let prods = std::iter::zip(&self.weights, xs) //zips weights, inputs together
        .map(|(a, b)| a*b) //maps to a single value
        .collect::<Vec<Value>>(); //collects into a Vec<Value>

        //let sum = prods.into_iter().fold(self.bias.clone(), |a, b| a + b);
         let sum = self.bias.clone() + prods.into_iter().reduce(|a, b| a + b).unwrap();
        //sum.relu()
        sum.tanh()
    }

    pub fn parameters(&self) -> Vec<Value> {
        [self.bias.clone()]
        .into_iter()
        .chain(self.weights.clone())
        .collect::<Vec<Value>>()
    }
}

// class Neuron(Module):

//     def __init__(self, nin, nonlin=True):
//         self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
//         self.b = Value(0)
//         self.nonlin = nonlin

//     def __call__(self, x):
//         act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
//         return act.relu() if self.nonlin else act

//     def parameters(self):
//         return self.w + [self.b]

//     def __repr__(self):
//         return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"