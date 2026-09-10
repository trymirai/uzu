use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use objc2_foundation::NSUInteger;

use crate::backends::apple::AppleEventSink;

pub type AppleEventRegistry = Arc<Mutex<HashMap<NSUInteger, AppleEventSink>>>;
