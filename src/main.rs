use std::io;

fn main() {
    let mut id = Vec::new();
    let mut buffer = String::new();
    let stdin = io::stdin();
    let _ = stdin.read_line(&mut buffer);
    let in_vec: Vec<char> = buffer.chars().collect();
    for i in in_vec {
        match i {
            'a' => id.push(1.0),
            'b' => id.push(2.0),
            'c' => id.push(3.0),
            'd' => id.push(4.0),
            'e' => id.push(5.0),
            'f' => id.push(6.0),
            'g' => id.push(7.0),
            'h' => id.push(8.0),
            'i' => id.push(9.0),
            'j' => id.push(10.0),
            'k' => id.push(11.0),
            'l' => id.push(12.0),
            'm' => id.push(13.0),
            'n' => id.push(14.0),
            'o' => id.push(15.0),
            'p' => id.push(16.0),
            'q' => id.push(17.0),
            'r' => id.push(18.0),
            's' => id.push(19.0),
            't' => id.push(20.0),
            'u' => id.push(21.0),
            'v' => id.push(22.0),
            'w' => id.push(23.0),
            'x' => id.push(24.0),
            'y' => id.push(25.0),
            'z' => id.push(26.0),
            _ => id.push(0.0),
        }
    }
    let mut temp = Vec::new();
    let mut other = Vec::new();
    for i in 0..id.len() - 1 {
        temp.push(id[i]);
        temp.push(id[i + 1]);
        other.push(temp.clone());
        temp.clear();
    }
    println!("{:?}", other)
}
