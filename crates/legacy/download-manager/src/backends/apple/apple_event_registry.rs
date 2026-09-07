use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::backends::apple::{AppleEventSink, AppleSinkKey};

pub type AppleEventRegistry = Arc<Mutex<HashMap<AppleSinkKey, AppleEventSink>>>;
