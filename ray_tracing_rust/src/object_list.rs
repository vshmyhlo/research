use crate::intersection::Intersection;
use crate::material::Material;
use crate::object::Object;
use crate::ray::Ray;

pub struct ObjectList {
    objects: Vec<Box<Object>>
}


impl ObjectList {
    pub fn new(objects: Vec<Box<Object>>) -> Self {
        ObjectList { objects }
    }


    pub fn intersects(&self, ray: &Ray) -> Option<Intersection> {
        self.objects
            .iter()
            .filter_map(|o| o.intersects(ray).map(|t| Intersection::new(&o, t)))
            .filter(|i| i.t >= 0.001)
            .min_by(|a, b| a.t.partial_cmp(&b.t).unwrap())
    }
}

