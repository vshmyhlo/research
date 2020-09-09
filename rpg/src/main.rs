trait MyTrait {
    fn m(&self);
}

struct S1 {}

impl S1 {
    pub fn new() -> Self {
        Self{}
    }
}

impl MyTrait for S1 {
    fn m(&self) {
        println!("hello S1");
    }
}

struct S2 {}

impl S2 {
    pub fn new() -> Self {
        Self{}
    }
}
impl MyTrait for S2 {
    fn m(&self) {
        println!("hello S2");
    }
}

struct D {}

impl Drop for D {
    fn drop(&mut self) {
        println!("drop D")
    }
}

fn main() {
    let v: Vec<Box<MyTrait>> = vec![
        Box::new(S1::new()),
        Box::new(S2::new()),
    ];

    for x in v {
        x.m();
    }

    let a = D {};
    let b = Box::new(a);
    let c  = b;

//    a;
//    b;
    c;
}
