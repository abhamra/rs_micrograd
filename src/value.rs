use std::{
    cell::{Ref, RefCell},
    collections::HashSet,
    convert::{From, Into},
    fmt::Debug,
    hash::Hash,
    hash::Hasher,
    iter::Sum,
    fmt::{Formatter, Result},
    ops::{Add, Deref, Mul, Neg, Sub},
    rc::Rc,
};

//create a value that can be referenced multiple times, 
//and subsequently owned multiple times! This enables us to
//work with a computational flow graph.
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value(Rc<RefCell<ValueNode>>);

impl Value {
    pub fn from<T>(t:T) -> Value 
        where T:Into<Value>, {
            t.into()
        }
    
    //constructor for the value object
    fn new(value: ValueNode) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    //accessor methods for Value attributes
    //self.borrow() accesses the Rc, then you can use
    // .{attribute} to access your information

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.borrow().grad
    }

    //clears gradients
    pub fn zero_grad(&self) {
        self.borrow_mut().grad = 0.0;
    }

    pub fn label(self, label: &str) -> Value {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    //a part of the backpropagation cycle, after generating
    //gradients, update the data by shifting it over per
    //the gradient's influence, using the learning rate
    //as a tunable parameter.
    pub fn update(&self, learning_rate:f64) {
        let mut value = self.borrow_mut();
        value.data += learning_rate * value.grad;
    }

    //gradients for functions below!
    pub fn pow(&self, other: &Value) -> Value {
        let result = self.borrow().data.powf(other.borrow().data);

        //define backward for the power node! 
        let backward:BackwardFn = |value| {
            let mut base = value.prev[0].borrow_mut();
            let power = value.prev[1].borrow();
            base.grad += power.data * (base.data.powf(power.data - 1.0)) * value.grad;
        };

        Value::new(ValueNode::new(
            result, //data
            None, //label
            Some("^".to_string()), //operation
            vec![self.clone(), other.clone()], //children, two outputs!
            Some(backward), //backward function
        ))
    }

    pub fn tanh(&self) -> Value {
        let result = self.borrow().data.tanh();

        //def backward for tanh node
        let backward:BackwardFn = |value| {
            let mut prev = value.prev[0].borrow_mut();
            prev.grad += (1.0 - value.data.powf(2.0)) * value.grad;
        };

        Value::new(ValueNode::new(
            result,
            None,
            Some("tanh".to_string()),
            vec![self.clone()], //only operates on one value, so one child
            Some(backward),
        ))
    }

    pub fn relu(&self) -> Value {
        let result = self.borrow().data.max(0.0);

        let backward:BackwardFn = |value| {
            let mut prev = value.prev[0].borrow_mut();
            let deriv:f64 = if value.data > 0.0 {1.0} else {0.0};
            prev.grad +=  deriv * value.grad;
        };

        Value::new(ValueNode::new(
            result,
            None,
            Some("relu".to_string()),
            vec![self.clone()], //only operates on one value, so one child
            Some(backward),
        ))
    }

    pub fn backward(&self) {
        let mut visited: HashSet<Value> = HashSet::new();

        self.borrow_mut().grad = 1.0;
        self.backward_internal(&mut visited, self);
    }

    fn backward_internal(&self, visited: &mut HashSet<Value>, value: &Value) {
        if !visited.contains(&value) {
            visited.insert(value.clone());

            let borrowed_value = value.borrow();
            if let Some(prop_fn) = borrowed_value.backward {
                prop_fn(&borrowed_value);
            }

            for child_id in &value.borrow().prev {
                self.backward_internal(visited, child_id);
            }
        }
    }

    // pub fn backward(&self) {
    //     let mut topo = Vec::new();
    //     let mut visited: HashSet<Value> = HashSet::new();

    //     fn build_topo(v: Value) {
    //         if !visited.contains(v) {
    //             visited.insert(value.clone());
    //         }
    //         for child in v.prev {
    //             build_topo(child);
    //         }
    //         topo.push(v)
    //     } //end build_topo
    //     build_topo(self);

    //     self.borrow().grad = 1.0;
    //     for v in topo.reverse() {
    //         self.backward(visited, v);
    //     }
    // }

}
//implementing Add and other math operations!

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other)
    }
}

fn add(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data + b.borrow().data;

    let backward:BackwardFn = |value| {
        let mut first = value.prev[0].borrow_mut();
        let mut second = value.prev[1].borrow_mut();

        first.grad += value.grad;
        second.grad += value.grad;
    };

    Value::new(ValueNode::new(
        result, 
        None, 
        Some("+".to_string()),
        vec![a.clone(), b.clone()],
        Some(backward),
    ))
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        add(&self, &(-other))
    }
}

impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;

    fn sub(self, other: &'b Value) -> Self::Output {
        add(self, &(-other))
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        mul(self, other)
    }
}

fn mul(a: &Value, b: &Value) -> Value {
    let result = a.borrow().data * b.borrow().data;

    let prop_fn: BackwardFn = |value| {
        let mut first = value.prev[0].borrow_mut();
        let mut second = value.prev[1].borrow_mut();

        first.grad += second.data * value.grad;
        second.grad += first.data * value.grad;
    };

    Value::new(ValueNode::new(
        result,
        None,
        Some("*".to_string()),
        vec![a.clone(), b.clone()],
        Some(prop_fn),
    ))
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(&self, &Value::from(-1))
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(self, &Value::from(-1))
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let mut sum = Value::from(0.0);
        loop {
            let val = iter.next();
            if val.is_none() {
                break;
            }

            sum = sum + val.unwrap();
        }
        sum
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t:T) -> Value {
        Value::new(ValueNode::new(t.into(), None, None, Vec::new(), None))
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueNode>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

type BackwardFn = fn(value: &Ref<ValueNode>);

pub struct ValueNode{
    data: f64,
    grad: f64,
    label: Option<String>,
    op: Option<String>,
    prev: Vec<Value>,
    backward: Option<BackwardFn>
}

impl ValueNode {
    fn new(
        data: f64,
        label: Option<String>,
        operation: Option<String>,
        previous: Vec<Value>,
        backward: Option<BackwardFn>,
    ) -> ValueNode {
        //returns new ValueNode
        ValueNode {
            data, 
            grad: 0.0,
            label, 
            op: operation,
            prev: previous,
            backward,
        }
    }
} 

//equivalence methods
impl Eq for ValueNode {}

impl PartialEq for ValueNode {
    fn eq(&self, other: &Self) -> bool
    {
        self.data == other.data
        && self.grad == other.grad
        && self.prev == other.prev
        && self.op == other.op
        && self.label == other.label
    }
}

//implementing Hash for HashSet (for visited set)
impl Hash for ValueNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.grad.to_bits().hash(state);
        self.op.hash(state);
        self.prev.hash(state);
        self.label.hash(state);
    }
}

impl Debug for ValueNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        //use &self.{element} to borrow
        f.debug_struct("ValueNode")
        .field("data", &self.data)
        .field("grad", &self.grad)
        .field("op", &self.op)
        .field("prev", &self.prev)
        .field("label", &self.label)
        .finish()
    }
}
