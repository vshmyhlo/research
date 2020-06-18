extern crate num;

use std::ops::Mul;

use image::Rgb;
use num::{Float, Zero};

use crate::traits::{CastU8, Round};
use crate::vector::Vector3;

pub type Color<T> = Vector3<T>;

impl<T> Color<T> where T: Zero {
    pub fn black() -> Self {
        Self::zero()
    }
}


impl<T> Color<T> where T: Float + From<u8> {
    pub fn to_rgb(self) -> Vector3<u8> {
        (self * 255_u8.into()).round().cast_u8()
    }
}


impl Color<u8> {
    pub fn to_pixel(self) -> Rgb<u8> {
        Rgb([1, 2, 3])
    }
}

