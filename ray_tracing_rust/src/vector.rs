extern crate num;

use std::ops::{Add, AddAssign, Div, Mul, Sub};

use image::Primitive;
use num::{Float, NumCast, Zero};

#[derive(Copy, Clone)]
pub struct Vector3<T> {
    x: T,
    y: T,
    z: T,
}

impl<T> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self {
            x: x,
            y: y,
            z: z,
        }
    }
}


impl<T> Vector3<T> where T: Float {
    pub fn round(self) -> Self {
        Self {
            x: self.x.round(),
            y: self.y.round(),
            z: self.z.round(),
        }
    }

    pub fn normalize(self) -> Self {
        self / self.norm()
    }

    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn norm(self) -> T {
        self.dot(self).sqrt()
    }
}

impl<T> Vector3<T> where T: Float, u8: NumCast {
    pub fn cast_u8(self) -> Vector3<u8> {
        Vector3 {
            x: NumCast::from(self.x).unwrap(),
            y: NumCast::from(self.y).unwrap(),
            z: NumCast::from(self.z).unwrap(),
        }
    }
}

impl<T> Add for Vector3<T> where T: Add<Output=T> {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}


impl<T> Sub<Self> for Vector3<T> where T: Sub<T, Output=T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T> Sub<T> for Vector3<T> where T: Sub<T, Output=T> + Copy {
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        Self {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
        }
    }
}

impl<T> Mul<Self> for Vector3<T> where T: Mul<T, Output=T> + Copy {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl<T> Mul<T> for Vector3<T> where T: Mul<T, Output=T> + Copy {
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<T> Div<T> for Vector3<T> where T: Div<T, Output=T> + Copy {
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl<T> AddAssign for Vector3<T> where T: AddAssign {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T> Zero for Vector3<T> where T: Zero {
    fn zero() -> Self {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        return self.x.is_zero() && self.y.is_zero() && self.z.is_zero();
    }
}


