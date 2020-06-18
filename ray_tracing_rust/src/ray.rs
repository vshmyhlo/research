use num::Float;

use crate::vector::Vector3;

pub struct Ray<T> {
    origin: Vector3<T>,
    direction: Vector3<T>,
}


impl<T> Ray<T> where T: Float {
    pub fn new(origin: Vector3<T>, direction: Vector3<T>) -> Self {
        Self {
            origin: origin,
            direction: direction.normalize(),
        }
    }
}

