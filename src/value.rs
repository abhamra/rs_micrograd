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
pub struct Value(Rc<RefCell<ValueNode>>);

impl Value {
    pub fn from<T>(t:T) -> Value 
        where T:Into<Value>, {
            t.into();
        }
    
    //constructor for the value object
    fn new(value: ValueNode) -> Value {
        Value(Rc::new(RefCell::new(value)));
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

    pub fn with_label(self, label: &str) -> Value {
        self.borrow_mut().label = Some(label.to_string());
        self
    }

    //a part of the backpropagation cycle, after generating
    //gradients, update the data by shifting it over per
    //the gradient's influence, using the learning rate
    //as a tunable parameter.
    pub fn update(&self, learning_rate:f64) {
        let mut value = self.borrow();
        value.data += learning_rate * value.grad;
    }

    //gradients for functions below!
    pub fn pow(&self, other: &Value) -> Value {
        let result = self.borrow().data.powf(other.borrow().data);

        //define backward for a node, then 
        
    }

}


impl Add for Value {
    type Output = Value;

    fn add(self, other:Self::Value) -> Self::Value {
        add(&self, &other)
    }
}


type BackwardFn = fn(value: &Ref<ValueNode>);

pub struct ValueNode{
    data: f64,
    grad: f64,
    label: Option<String>,
    op: Option<String>,
    prev: Vec<Value>,
    backward: Option<Backward>
}

impl ValueNode {
    fn new(
        data: f64,
        label: Option<String>,
        operation: Option<String>,
        previous: Vec<Value>,
        back: Option<BackwardFn>,
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
    }
}