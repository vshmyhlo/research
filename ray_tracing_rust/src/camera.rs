use crate::ray::Ray;
use crate::vector::Vector3;

pub struct Camera {
    origin: Vector3,
}

impl Camera {
    pub fn new(origin: Vector3) -> Self {
        Self { origin }
    }
}

impl Camera {
    pub fn position_ray(&self, u: f32, v: f32) -> Ray {
        let pixel_position = Vector3 {
            x: u * 2_f32 - 1_f32,
            y: v * 2_f32 - 1_f32,
            z: 0_f32,
        };

        Ray::new(self.origin, pixel_position - self.origin)
    }
}
