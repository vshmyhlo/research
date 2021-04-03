use crate::color::Color;
use crate::ray::Ray;
use crate::reflection::Reflection;
use crate::vector::Vector3;

pub trait Material {
    fn reflect(&self, ray: &Ray, t: f32, normal: Vector3) -> Option<Reflection>;
    fn emit(&self) -> Color;
}

pub struct Light {
    color: Color,

    _secret: (),
}

impl Light {
    pub fn new(color: Color) -> Self {
        Self { color, _secret: () }
    }
}

impl Material for Light {
    fn reflect(&self, ray: &Ray, t: f32, normal: Vector3) -> Option<Reflection> {
        None
    }

    fn emit(&self) -> Color {
        self.color
    }
}

pub struct Metal {
    color: Color,

    _secret: (),
}

impl Metal {
    pub fn new(color: Color) -> Self {
        Self { color, _secret: () }
    }
}

impl Material for Metal {
    fn reflect(&self, ray: &Ray, t: f32, normal: Vector3) -> Option<Reflection> {
        let reflected = Vector3::reflect(ray.direction, normal);
        let reflected = Ray::new(ray.position_at(t), reflected);

        if reflected.direction.dot(normal) <= 0. {
            return None;
        }

        Some(Reflection::new(reflected, self.color))
    }

    fn emit(&self) -> Color {
        Color::black()
    }
}

pub struct Diffuse {
    color: Color,

    _secret: (),
}

impl Diffuse {
    pub fn new(color: Color) -> Self {
        Self { color, _secret: () }
    }
}

impl Material for Diffuse {
    fn reflect(&self, ray: &Ray, t: f32, normal: Vector3) -> Option<Reflection> {
        let ray = Ray::new(ray.position_at(t), Vector3::random_in_hemisphere(normal));

        Some(Reflection::new(ray, self.color))
    }

    fn emit(&self) -> Color {
        Color::black()
    }
}
