//! Inspect safetensors weight file structure

use std::path::PathBuf;

use burn::store::{ModuleStore, SafetensorsStore};

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Volumes/tb3ssd/volume/burn_models/qwen3_4b_text_encoder.safetensors"));

    eprintln!("Inspecting: {:?}", path);

    let mut store = SafetensorsStore::from_file(&path);

    match store.keys() {
        Ok(keys) => {
            println!("Total keys: {}", keys.len());
            println!();

            // Print first 30 keys
            for key in keys.iter().take(30) {
                if let Ok(Some(snapshot)) = store.get_snapshot(key) {
                    if let Ok(data) = snapshot.to_data() {
                        println!("{}: {:?} dtype={:?}", key, data.shape, data.dtype);
                    } else {
                        println!("{}: (could not read data)", key);
                    }
                } else {
                    println!("{}: (no snapshot)", key);
                }
            }

            if keys.len() > 30 {
                println!("... ({} more)", keys.len() - 30);
            }
        }
        Err(e) => {
            eprintln!("Error getting keys: {:?}", e);
        }
    }
}
