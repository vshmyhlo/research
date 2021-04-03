extern crate clap;
extern crate image;
extern crate num;
extern crate rand;

use clap::clap_app;
use clap::{App, Arg};
use image::{ImageBuffer, RgbImage};
use rand::Rng;

use camera::Camera;
use color::Color;
use object::Sphere;
use object_list::ObjectList;
use ray::Ray;
use vector::Vector3;

use crate::material::{Diffuse, Light, Metal};
use crate::object::Object;

mod camera;
mod color;
mod intersection;
mod material;
mod object;
mod object_list;
mod ray;
mod reflection;
mod traits;
mod vector;

const BACKGROUND_COLOR: Color = Vector3 {
    x: 0.4,
    y: 0.4,
    z: 0.4,
};

fn main() {
    let matches = clap_app!(app =>
        (@arg size: --size +takes_value)
        (@arg rep: --rep +takes_value)
    )
    .get_matches();

    let size: u32 = matches.value_of_t("size").unwrap();
    let rep: u32 = matches.value_of_t("rep").unwrap();

    let mut image: RgbImage = ImageBuffer::new(size, size);
    let mut rng = rand::thread_rng();
    let camera = Camera::new(Vector3 {
        x: 0.,
        y: 0.,
        z: -1.,
    });

    let objects: Vec<Box<Object>> = vec![
        Box::new(Sphere::new(
            Vector3 {
                x: 0.,
                y: -101.,
                z: 5.,
            },
            100.,
            Diffuse::new(Vector3 {
                x: 1.,
                y: 1.,
                z: 0.,
            }),
        )),
        Box::new(Sphere::new(
            Vector3 {
                x: -2.5,
                y: 0.,
                z: 5.,
            },
            1.,
            Light::new(Vector3 {
                x: 0.,
                y: 1.,
                z: 1.,
            }),
        )),
        Box::new(Sphere::new(
            Vector3 {
                x: 2.5,
                y: 0.,
                z: 5.,
            },
            1.,
            Light::new(Vector3 {
                x: 1.,
                y: 1.,
                z: 1.,
            }),
        )),
        Box::new(Sphere::new(
            Vector3 {
                x: 0.,
                y: 0.,
                z: 5.,
            },
            1.,
            Metal::new(Vector3 {
                x: 0.5,
                y: 0.5,
                z: 0.5,
            }),
        )),
        Box::new(Sphere::new(
            Vector3 {
                x: 0.,
                y: 2.5,
                z: 5.,
            },
            1.,
            Diffuse::new(Vector3 {
                x: 1.,
                y: 0.,
                z: 1.,
            }),
        )),
    ];
    let objects = ObjectList::new(objects);

    for x in 0..size {
        for y in 0..size {
            let mut color = Color::black();
            for _ in 0..rep {
                let u = (x as f32 + rng.gen::<f32>()) / (size as f32);
                let v = (y as f32 + rng.gen::<f32>()) / (size as f32);

                let ray = camera.position_ray(u, v);
                color += trace_ray(&ray, &objects, 32)
            }

            image[(size - x - 1, size - y - 1)] = (color / (rep as f32)).to_rgb_pixel();
        }
    }

    image.save("./data/output.png").unwrap();
}

fn trace_ray(ray: &Ray, objects: &ObjectList, max_depth: u32) -> Color {
    if max_depth == 0 {
        return Color::black();
    }

    match objects.intersects(ray) {
        None => BACKGROUND_COLOR,
        Some(intersection) => {
            let position = ray.position_at(intersection.t);
            let normal = intersection.object.normal_at(position);

            let emitted = intersection.object.get_material().emit();
            let reflection =
                intersection
                    .object
                    .get_material()
                    .reflect(ray, intersection.t, normal);

            match reflection {
                None => emitted,
                Some(reflection) => {
                    emitted
                        + reflection.attenuation
                            * trace_ray(&reflection.ray, objects, max_depth - 1)
                }
            }
        }
    }
}
