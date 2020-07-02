extern crate num;

use std::fmt::Display;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use num::Zero;
use rand::distributions::Uniform;
use rand::Rng;

#[derive(Debug, Copy, Clone)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,

}


impl Vector3 {
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

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn reflect(v: Vector3, n: Vector3) -> Vector3 {
        v - n * 2. * v.dot(n)
    }


    pub fn random_unit() -> Self {
        let mut rng = rand::thread_rng();
        let dist = Uniform::new(-1_f32, 1_f32);

        Vector3 {
            x: rng.sample(dist),
            y: rng.sample(dist),
            z: rng.sample(dist),
        }.normalize()
    }


    pub fn random_in_hemisphere(n: Vector3) -> Vector3 {
        let v = Self::random_unit();

        if v.dot(n) > 0.0 {
            return v;
        } else {
            return -v;
        }
    }
}


impl Add<Self> for Vector3 {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Add<f32> for Vector3 {
    type Output = Self;

    fn add(self, other: f32) -> Self::Output {
        Self {
            x: self.x + other,
            y: self.y + other,
            z: self.z + other,
        }
    }
}


impl Sub<Self> for Vector3 {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Sub<f32> for Vector3 {
    type Output = Self;

    fn sub(self, other: f32) -> Self::Output {
        Self {
            x: self.x - other,
            y: self.y - other,
            z: self.z - other,
        }
    }
}

impl Mul<Self> for Vector3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl Mul<f32> for Vector3 {
    type Output = Self;

    fn mul(self, other: f32) -> Self::Output {
        Self {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl Div<f32> for Vector3 {
    type Output = Self;

    fn div(self, other: f32) -> Self::Output {
        Self {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

impl AddAssign for Vector3 {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl Neg for Vector3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Zero for Vector3 {
    fn zero() -> Self {
        Self {
            x: f32::zero(),
            y: f32::zero(),
            z: f32::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        return self.x.is_zero() && self.y.is_zero() && self.z.is_zero();
    }
}


