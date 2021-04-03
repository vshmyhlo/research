use crate::material::{Light, Material};
use crate::ray::Ray;
use crate::vector::Vector3;

pub trait Object {
    fn normal_at(&self, position: Vector3) -> Vector3;
    fn get_material(&self) -> &dyn Material;
    fn intersects(&self, ray: &Ray) -> Option<f32>;
}

pub struct Sphere<M>
where
    M: Material,
{
    center: Vector3,
    radius: f32,
    material: M,
}

impl<M> Sphere<M>
where
    M: Material,
{
    pub fn new(center: Vector3, radius: f32, material: M) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl<M> Object for Sphere<M>
where
    M: Material,
{
    fn normal_at(&self, position: Vector3) -> Vector3 {
        (position - self.center).normalize()
    }

    fn get_material(&self) -> &dyn Material {
        &self.material
    }

    fn intersects(&self, ray: &Ray) -> Option<f32> {
        let sr = ray.origin - self.center;
        let a = ray.direction.dot(ray.direction);
        let b = 2. * ray.direction.dot(sr);
        let c = sr.dot(sr) - self.radius.powi(2);

        let disc = b.powi(2) - 4. * a * c;
        if disc < 0. {
            return None;
        }

        let t = (-b - disc.sqrt()) / (2. * a);

        if t <= 0. {
            return None;
        }

        Some(t)
    }

    //    sr = ray.origin - self.center
    //
    //    a = torch.dot(ray.direction, ray.direction)
    //    b = 2 * torch.dot(ray.direction, sr)
    //    c = torch.dot(sr, sr) - self.radius**2
    //
    //    disc = b**2 - 4 * a * c
    //    if disc < 0:
    //    return None
    //
    //    t = (-b - torch.sqrt(disc)) / (2 * a)
    //    if t <= 0:
    //    return None
    //
    //    return t
}
