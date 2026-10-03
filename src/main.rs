use std::io;

use crate::ai::embedding::Embedding;
mod ai;

fn main() {
    let mut id: Vec<usize> = Vec::new();
    let mut buffer = String::new();
    let stdin = io::stdin();
    let _ = stdin.read_line(&mut buffer);
    buffer = buffer.trim().to_string();
    let in_vec: Vec<char> = buffer.chars().collect();
    for i in in_vec {
        match i {
            'a' => id.push(1),
            'b' => id.push(2),
            'c' => id.push(3),
            'd' => id.push(4),
            'e' => id.push(5),
            'f' => id.push(6),
            'g' => id.push(7),
            'h' => id.push(8),
            'i' => id.push(9),
            'j' => id.push(10),
            'k' => id.push(11),
            'l' => id.push(12),
            'm' => id.push(13),
            'n' => id.push(14),
            'o' => id.push(15),
            'p' => id.push(16),
            'q' => id.push(17),
            'r' => id.push(18),
            's' => id.push(19),
            't' => id.push(20),
            'u' => id.push(21),
            'v' => id.push(22),
            'w' => id.push(23),
            'x' => id.push(24),
            'y' => id.push(25),
            'z' => id.push(26),
            _ => id.push(0),
        }
    }
    // let mut temp = Vec::new();
    // let mut other = Vec::new();
    // for i in 0..id.len() - 1 {
    //     temp.push(id[i]);
    //     temp.push(id[i + 1]);
    //     other.push(temp.clone());
    //     temp.clear();
    // }
    // for arr in other {
    //     let input = arr[0];
    //     let target = arr[1];
    //     println!("input : {input}");
    //     println!("target : {target}");
    // }

    let mut w = Vec::new();
    for _ in 0..27 {
        let b = vec![
            rand::random_range(0.0..1.0),
            rand::random_range(0.0..1.0),
            rand::random_range(0.0..1.0),
            rand::random_range(0.0..1.0),
        ];
        w.push(b);
    }
    for i in 0..w.len() {
        println!("{:?}", w[i]);
    }

    // let embedding = Embedding { weights: a };
    // let output = embedding.forward(&id);
    // println!("em output : {:?}", output);
}
