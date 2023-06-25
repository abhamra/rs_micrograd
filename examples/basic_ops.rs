use rs_micrograd::Value;

fn main() {
    let a = Value::from(2.0).label("a");
    let a_c = a.clone();
    let b = Value::from(3.0).label("b");
    let b_c = b.clone();
    let c = Value::from(4.0).label("c");
    let c_c = c.clone();
    let ab = Value::from(a*b).label("ab");
    let output = Value::from(ab + c).label("output");

    output.backward();

    assert_eq!(a_c.grad(), b_c.data());
    assert_eq!(c_c.grad(), 1.0);
    println!("-----");
    println!("{:?}", output);
}