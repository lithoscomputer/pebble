//! The pebble command line as a library.
//!
//! The `pebble` binary is a thin argument parser over this crate. Another
//! program that wants the same sessions with its own front end takes what it
//! needs: [`session::run_prompt`] runs one prompt on an agent it built, with
//! the [`render`] module's event renderer and closing summary and the
//! [`approval`] module's terminal prompt for tools the permission level does
//! not allow outright; [`exec`], [`auth`], and [`interactive`] are the
//! binary's commands whole, arguments included.

pub mod application;
pub mod approval;
pub mod auth;
pub mod credentials;
pub mod exec;
pub mod interactive;
pub mod render;
pub mod resources;
pub mod secret_input;
pub mod session;
pub mod settings;
pub mod storage;
pub mod terminal;
