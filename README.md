# qwen3-burn

Qwen3 language model implementation using the [Burn](https://burn.dev/) deep learning framework in Rust.

## Features

- Qwen3 decoder-only transformer architecture (GQA, RoPE, RMSNorm, SwiGLU, QK-norm)
- Text generation with temperature, top-k, and top-p sampling
- KV cache for efficient autoregressive generation
- Load pretrained weights from HuggingFace safetensors format
- Convert weights to Burn's `.bpk` format
- Tokenizer support via the `tokenizers` crate
- Compatible with Burn backends: CPU (NdArray), Metal (Candle), and more

## Supported Model Sizes

| Model | Layers | Hidden | Heads | KV Heads |
|-------|--------|--------|-------|----------|
| 0.6B  | 28     | 1024   | 16    | 8        |
| 1.7B  | 28     | 2048   | 16    | 8        |
| 4B    | 36     | 2560   | 32    | 8        |
| 8B    | 36     | 4096   | 32    | 8        |
| 14B   | 40     | 5120   | 40    | 8        |
| 32B   | 64     | 5120   | 40    | 8        |

## Usage

### Prerequisites

Download a Qwen3 model (e.g. [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)) and place `model.safetensors` and `tokenizer.json` in a `models/` directory.

### Text Generation (CPU)

```bash
cargo run --example generate --release -- \
  --model models/model.safetensors \
  --tokenizer models/tokenizer.json \
  --prompt "Hello, I am a language model and" \
  --max-tokens 50
```

### Text Generation (Metal / Apple Silicon)

```bash
cargo run --example generate-metal --release --features candle -- \
  --model models/model.safetensors \
  --tokenizer models/tokenizer.json \
  --prompt "Hello, I am a language model and" \
  --max-tokens 50
```

### Convert Weights to Burn .bpk Format

```bash
cargo run --example convert_to_bpk --release -- \
  --input models/model.safetensors \
  --output models/model.bpk
```

### As a Library

```rust
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};

// Load tokenizer and model
let tokenizer = Qwen3Tokenizer::from_file("tokenizer.json")?;
let device = Default::default();
let config = Qwen3Config::qwen3_0_6b();
let mut model: Qwen3ForCausalLM<Backend> = config.init_causal_lm(&device);
model.load_weights("model.safetensors")?;

// Tokenize and generate
let (input_ids, _) = tokenizer.encode("Once upon a time")?;
let input_tensor = Tensor::from_data(input_ids.as_slice(), &device).unsqueeze();
let output = model.generate_with_cache(input_tensor, 50, 0.7, 0.9, 50);

// Decode
let output_tokens: Vec<u32> = output.to_data().to_vec()?
    .iter().map(|&x: &i64| x as u32).collect();
let text = tokenizer.decode(&output_tokens)?;
```

## Project Structure

```
src/
  lib.rs        - Public API and module exports
  decoder.rs    - Qwen3Model, Qwen3ForCausalLM, config presets
  attention.rs  - Grouped Query Attention with QK-norm
  rope.rs       - Rotary Position Embeddings
  cache.rs      - KV cache for autoregressive generation
  load.rs       - Safetensors weight loading
  tokenizer.rs  - Tokenizer wrapper
examples/
  generate.rs       - CPU text generation
  generate-metal.rs - Metal-accelerated text generation
  convert_to_bpk.rs - Weight format conversion
  inspect_weights.rs - Weight inspection utility
```

## License

MIT OR Apache-2.0
