//! Example: Text generation with Qwen3-0.6B using Metal backend
//!
//! Usage:
//!   cargo run --example generate-metal --release -- --model models/model.safetensors --tokenizer models/tokenizer.json

use std::path::PathBuf;

use burn::backend::candle::{Candle, CandleDevice};
use burn::tensor::{Int, Tensor};
use half::bf16;
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};

type Backend = Candle<bf16, i64>;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let model_path = args.iter()
        .position(|x| x == "--model")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("models/model.safetensors"));

    let tokenizer_path = args.iter()
        .position(|x| x == "--tokenizer")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("models/tokenizer.json"));

    let prompt = args.iter()
        .position(|x| x == "--prompt")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "Hello, I am a language model and".to_string());

    let max_tokens: usize = args.iter()
        .position(|x| x == "--max-tokens")
        .map(|i| args[i + 1].parse().unwrap_or(50))
        .unwrap_or(50);

    println!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)?;

    println!("Initializing Qwen3-0.6B model on Metal...");
    let device = CandleDevice::metal(0);

    // Use the 0.6B config preset
    let config = Qwen3Config::qwen3_0_6b();
    println!("Config: {} layers, {} hidden, {} heads",
        config.num_hidden_layers, config.hidden_size, config.num_attention_heads);

    let mut model: Qwen3ForCausalLM<Backend> = config.init_causal_lm(&device);

    println!("Loading weights from {:?}...", model_path);
    model.load_weights(&model_path).map_err(|e| format!("Failed to load weights: {e:?}"))?;
    println!("Model loaded successfully!");

    // Tokenize prompt (don't pad to max_length for generation)
    println!("\nPrompt: {}", prompt);
    let tokenizer = tokenizer.with_max_length(512);  // Set a reasonable max but don't force padding
    let encoding = tokenizer.encode(&prompt)?;
    // Take only the non-padding tokens
    let input_ids: Vec<i64> = encoding.0.iter()
        .zip(encoding.1.iter())
        .filter(|&(_, &mask)| mask)
        .map(|(&id, _)| id)
        .collect();
    println!("Input tokens ({}): {:?}", input_ids.len(), input_ids);

    // Create input tensor [1, seq_len]
    let input_tensor: Tensor<Backend, 1, Int> = Tensor::from_data(
        input_ids.as_slice(),
        &device,
    );
    let input_tensor: Tensor<Backend, 2, Int> = input_tensor.unsqueeze();

    println!("\nGenerating {} tokens...", max_tokens);
    let start = std::time::Instant::now();

    // Generate with KV cache for efficiency
    // Note: temperature=0 uses greedy decoding (fast), temperature>0 uses sampling (slower on Metal)
    let output = model.generate_with_cache(
        input_tensor,
        max_tokens,
        0.0,   // temperature (0 = greedy, >0 = sampling)
        0.9,   // top_p
        50,    // top_k
    );

    let elapsed = start.elapsed();

    // Get output tokens
    let output_data: Vec<i64> = output.to_data().to_vec()
        .map_err(|e| format!("Failed to convert tensor: {e:?}"))?;
    let output_tokens: Vec<u32> = output_data.iter().map(|&x| x as u32).collect();

    println!("Output tokens ({} total): {:?}", output_tokens.len(), &output_tokens[..output_tokens.len().min(30)]);

    // Decode
    let generated_text = tokenizer.decode(&output_tokens)?;

    println!("\n=== Generated Text ===");
    println!("{}", generated_text);
    println!("======================");

    let tokens_per_sec = max_tokens as f64 / elapsed.as_secs_f64();
    println!("\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
        max_tokens, elapsed.as_secs_f64(), tokens_per_sec);

    Ok(())
}
