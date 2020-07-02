use crate::ray::Ray;
use crate::vector::Vector3;

pub struct Reflection {
    pub ray: Ray,
    pub attenuation: Vector3,

    _secret: (),
}

impl Reflection {
    pub fn new(ray: Ray, attenuation: Vector3) -> Self {
        Self { ray, attenuation, _secret: () }
    }
}