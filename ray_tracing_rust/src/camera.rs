use std::ops::Mul;

use num::Float;

use crate::ray::Ray;
use crate::vector::Vector3;

pub struct Camera<T> {
    origin: Vector3<T>
}

impl<T> Camera<T> {
    pub fn new(origin: Vector3<T>) -> Self {
        Self {
            origin: origin
        }
    }
}


impl<T> Camera<T> where T: Float + From<f32> {
    pub fn position_ray(&self, u: T, v: T) -> Ray<T> {
        let position = Vector3::new(
            u * 2_f32.into() - 1_f32.into(),
            v * 2_f32.into() - 1_f32.into(),
            0_f32.into(),
        );
        Ray::new(self.origin, position - self.origin)
    }
}