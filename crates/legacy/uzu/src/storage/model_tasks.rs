use std::{collections::HashMap, sync::Arc};

use download_manager::DownloadTask;
use kiban::rt::TaskJoinHandle;
use shoji::types::model::ModelIdentifier;

pub type ModelTasks = HashMap<ModelIdentifier, (Arc<DownloadTask>, Box<dyn TaskJoinHandle<()>>)>;
