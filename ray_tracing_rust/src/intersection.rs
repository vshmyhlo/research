use crate::material::Material;
use crate::object::Object;

pub struct Intersection<'a> {
    pub object: &'a Box<Object>,
    pub t: f32,

    _secret: (),
}

impl<'a> Intersection<'a> {
    pub fn new(object: &'a Box<Object>, t: f32) -> Self {
        Intersection {
            object,
            t,
            _secret: (),
        }
    }
}