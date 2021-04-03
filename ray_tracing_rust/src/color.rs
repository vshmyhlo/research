extern crate num;

use image::Rgb;
use num::Zero;

use crate::vector::Vector3;

pub type Color = Vector3;

impl Color {
    pub fn black() -> Self {
        Self::zero()
    }
}

impl Color {
    pub fn to_rgb_pixel(self) -> Rgb<u8> {
        let rgb = (self * 255_f32).round();
        Rgb([rgb.x as u8, rgb.y as u8, rgb.z as u8])
    }
}
