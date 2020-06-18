extern crate clap;
extern crate image;
extern crate num;
extern crate rand;

use clap::{App, Arg};
use image::{ImageBuffer, Pixel, Rgb, RgbImage};
use num::NumCast;
use num::Zero;
use rand::Rng;

use camera::Camera;
use color::Color;
use object::ObjectList;
use ray::Ray;
use vector::Vector3;

mod ray;
mod traits;
mod vector;
mod object;
mod color;
mod camera;


type F = f32;

fn main() {
//    let a: f32 = 1.0;
//    let b: u8 = NumCast::from(a).unwrap();


    let matches = App::new("Ray Tracing")
        .arg(Arg::new("size")
            .long("size")
            .value_name("SIZE")
            .takes_value(true))
        .get_matches();


    let size: u32 = matches.value_of_t("size").unwrap();
    let max_steps = 32;
    let mut image: RgbImage = ImageBuffer::new(size, size);
    let mut rng = rand::thread_rng();
    let camera = Camera::new(Vector3::<F>::new(0 as F, 0 as F, -1 as F));
    let objects = ObjectList::new();


    for x in 0..size {
        for y in 0..size {
            let mut color = Color::black();
            for _ in 0..1 {
                let u = (x as f32 + rng.gen::<f32>()) / (size as f32);
                let v = (y as f32 + rng.gen::<f32>()) / (size as f32);

                let ray = camera.position_ray(u, v);
                color += trace_ray(ray, &objects, max_steps)
            }

            image[(x, y)] = color.to_rgb().to_pixel();
        }
    }


    image.save("./output.png").unwrap();
}


fn trace_ray(ray: Ray<f32>, objects: &ObjectList, max_depth: u32) -> Color<f32> {
    return Color::black();
}
