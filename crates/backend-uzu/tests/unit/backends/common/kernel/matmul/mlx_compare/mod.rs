#![cfg(backend = "metal")]

mod bench;
mod matrix;
mod mlx;
mod summary;
mod table;
mod uzu;

pub use matrix::Cell;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Uzu,
    Mlx,
}

pub trait Matmul {
    fn engine(&self) -> Engine;

    fn name(&self) -> &str;

    fn prepare(
        &mut self,
        cell: Cell,
    ) -> Result<(), String>;

    fn dispatch(
        &mut self,
        count: u64,
    ) -> Result<(), String>;
}
