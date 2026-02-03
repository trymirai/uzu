#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/dsl_structs.rs"));

include!(concat!(env!("OUT_DIR"), "/traits.rs"));

pub mod sampling;
