//! Qwen3 language model implementation using the Burn deep learning framework.
//!
//! This crate provides a Rust implementation of the Qwen3 decoder-only transformer
//! architecture. Qwen3 is a large language model developed by Alibaba that features:
//!
//! - Grouped Query Attention (GQA) for efficient KV caching
//! - RoPE (Rotary Position Embeddings) with theta=1,000,000
//! - RMSNorm for layer normalization
//! - SwiGLU activation in feed-forward layers
//! - QK normalization in attention
//!
//! # Models
//!
//! This crate provides two model variants:
//!
//! - [`Qwen3Model`] - Base transformer model for embeddings and hidden states
//! - [`Qwen3ForCausalLM`] - Full causal language model with generation capabilities
//!
//! # Features
//!
//! - Load pretrained weights from HuggingFace safetensors format
//! - Tokenizer support via the `tokenizers` crate
//! - Text generation with temperature, top-k, and top-p sampling
//! - Extract hidden states for text embeddings
//! - Compatible with all Burn backends (CPU, CUDA, Metal, etc.)
//!
//! # Text Generation Example
//!
//! ```ignore
//! use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};
//!
//! // Load tokenizer and model
//! let tokenizer = Qwen3Tokenizer::from_file("tokenizer.json")?;
//! let mut model = Qwen3Config::default().init_causal_lm::<Backend>(&device);
//! model.load_weights("model.safetensors")?;
//!
//! // Tokenize prompt
//! let (input_ids, _) = tokenizer.encode("Once upon a time")?;
//! let input_tensor = Tensor::from_data(&input_ids, &device).unsqueeze();
//!
//! // Generate text
//! let output = model.generate(
//!     input_tensor,
//!     50,    // max_new_tokens
//!     0.7,   // temperature
//!     0.9,   // top_p
//!     50,    // top_k
//! );
//!
//! // Decode output tokens
//! let generated_text = tokenizer.decode(output)?;
//! ```
//!
//! # Text Embedding Example
//!
//! ```ignore
//! use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};
//!
//! // Load tokenizer and base model
//! let tokenizer = Qwen3Tokenizer::from_file("tokenizer.json")?;
//! let mut model = Qwen3Config::default().init::<Backend>(&device);
//! model.load_weights("model.safetensors")?;
//!
//! // Get embeddings (second-to-last layer hidden states)
//! let (input_ids, attention_mask) = tokenizer.encode_prompt("Hello, world!")?;
//! let embeddings = model.encode(input_ids_tensor, attention_mask_tensor);
//! ```
//!
//! # Model Configurations
//!
//! The default configuration matches the Z-Image text encoder variant:
//! - 36 layers, 2560 hidden size, 32 attention heads, 8 KV heads
//!
//! Common Qwen3 model sizes:
//!
//! | Model | Layers | Hidden | Heads | KV Heads |
//! |-------|--------|--------|-------|----------|
//! | 0.6B  | 28     | 1024   | 16    | 8        |
//! | 1.7B  | 28     | 2048   | 16    | 8        |
//! | 4B    | 36     | 2560   | 32    | 8        |
//! | 8B    | 36     | 4096   | 32    | 8        |
//! | 14B   | 40     | 5120   | 40    | 8        |
//! | 32B   | 64     | 5120   | 40    | 8        |

mod attention;
mod cache;
mod decoder;
pub mod load;
mod rope;
mod tokenizer;

pub use cache::{KVCache, ModelCache};
pub use decoder::{Qwen3Config, Qwen3ForCausalLM, Qwen3Model};
pub use tokenizer::Qwen3Tokenizer;
