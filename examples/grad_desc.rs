use rs_micrograd::{Value, MLP};

fn main() {
    let mlp = MLP::new(3, vec![4, 4, 1]);

    let xs = vec![
        vec![2.0, 3.0, -1.0],
        vec![3.0, -1.0, 0.5],
        vec![0.5, 1.0, 1.0],
        vec![1.0, 1.0, -1.0],
    ];

    let ys = vec![1.0, -1.0, -1.0, 1.0];

    let epochs = 100;
    let learning_rate = -0.02;

    for _ in 0..epochs {
        //forward pass
        let ypred: Vec<Value> = xs
        .iter()
        .map(|x| mlp.forward(x.iter().map(|x| Value::from(*x)).collect())[0].clone())
        .collect();
        let ypred_fl:Vec<f64> = ypred.iter().map(|x| x.data()).collect();

        //loss fn
        let ygt = ys.iter().map(|y| Value::from(*y)); //converting into an array of values
        let loss: Value = ypred.into_iter().zip(ygt)
        .map(|(yp, yg)| (yp - yg).pow(&Value::from(2.0)))
        .sum();

        println!("Loss: {} Predictions: {:?}", loss.data(), ypred_fl);

        //backward pass
        mlp.parameters().iter().for_each(|p| p.zero_grad());
        loss.backward();

        //update grads
        mlp.parameters().iter().for_each(|p| p.update(learning_rate));
    }
}

