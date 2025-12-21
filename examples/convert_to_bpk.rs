//! Convert Qwen3 safetensors weights to Burn .bpk format.
//!
//! Usage:
//!   cargo run --example convert_to_bpk --release -- --input model.safetensors --output model.bpk

use std::path::PathBuf;

use burn::backend::candle::{Candle, CandleDevice};
use burn::store::{BurnpackStore, ModuleStore};
use half::bf16;
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM};

// Use Candle with BF16 to match the model weights
type Backend = Candle<bf16, i64>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();

    let input_path = args.iter()
        .position(|x| x == "--input")
        .map(|i| PathBuf::from(&args[i + 1]))
        .ok_or("Missing --input argument")?;

    let output_path = args.iter()
        .position(|x| x == "--output")
        .map(|i| PathBuf::from(&args[i + 1]))
        .ok_or("Missing --output argument")?;

    // Determine model size from filename or use 0.6B as default
    let config = if input_path.to_string_lossy().contains("0.6") {
        println!("Using Qwen3-0.6B configuration");
        Qwen3Config::qwen3_0_6b()
    } else if input_path.to_string_lossy().contains("1.7") {
        println!("Using Qwen3-1.7B configuration");
        Qwen3Config::qwen3_1_7b()
    } else if input_path.to_string_lossy().contains("4b") || input_path.to_string_lossy().contains("4B") {
        println!("Using Qwen3-4B configuration");
        Qwen3Config::qwen3_4b()
    } else if input_path.to_string_lossy().contains("8b") || input_path.to_string_lossy().contains("8B") {
        println!("Using Qwen3-8B configuration");
        Qwen3Config::qwen3_8b()
    } else {
        println!("Using default Qwen3-0.6B configuration");
        Qwen3Config::qwen3_0_6b()
    };

    println!("Config: {} layers, {} hidden, {} heads, head_dim={:?}",
        config.num_hidden_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.head_dim);

    println!("\nInitializing model...");
    let device = CandleDevice::Cpu;
    let mut model: Qwen3ForCausalLM<Backend> = config.init_causal_lm(&device);

    println!("Loading weights from {:?}...", input_path);
    model.load_weights(&input_path)
        .map_err(|e| format!("Failed to load weights: {e:?}"))?;

    println!("Saving to {:?}...", output_path);
    let mut store = BurnpackStore::from_file(&output_path)
        .auto_extension(false);
    store.collect_from(&model)
        .map_err(|e| format!("Failed to save: {e:?}"))?;

    // Get file size
    let metadata = std::fs::metadata(&output_path)
        .map_err(|e| format!("Failed to get file info: {e}"))?;
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);

    println!("\nConversion complete!");
    println!("Output: {:?} ({:.1} MB)", output_path, size_mb);

    Ok(())
}
