use crate::vector::Vector3;

pub struct Ray {
    pub origin: Vector3,
    pub direction: Vector3,

    _secret: (),
}

impl Ray {
    pub fn new(origin: Vector3, direction: Vector3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            _secret: (),
        }
    }

    pub fn position_at(&self, t: f32) -> Vector3 {
        self.origin + self.direction * t
    }
}
