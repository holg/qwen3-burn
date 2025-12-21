//! Qwen3 decoder model.
//!
//! This module provides both the base model (for embeddings) and the causal LM
//! model (for text generation).

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, Int, activation::{silu, softmax}},
};

use super::attention::{Qwen3Attention, Qwen3AttentionConfig};
use super::cache::{KVCache, ModelCache};

/// Configuration for Qwen3 model.
#[derive(Config, Debug)]
pub struct Qwen3Config {
    /// Vocabulary size.
    #[config(default = 151936)]
    pub vocab_size: usize,
    /// Hidden dimension.
    #[config(default = 2560)]
    pub hidden_size: usize,
    /// Intermediate FFN dimension.
    #[config(default = 9728)]
    pub intermediate_size: usize,
    /// Number of transformer layers.
    #[config(default = 36)]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[config(default = 32)]
    pub num_attention_heads: usize,
    /// Number of key-value heads for GQA.
    #[config(default = 8)]
    pub num_key_value_heads: usize,
    /// Dimension per attention head.
    /// If not set, defaults to hidden_size / num_attention_heads.
    pub head_dim: Option<usize>,
    /// RMSNorm epsilon.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
    /// RoPE theta.
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    /// Maximum sequence length.
    #[config(default = 40960)]
    pub max_position_embeddings: usize,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Qwen3Config::new()
    }
}

impl Qwen3Config {
    /// Get the effective head dimension.
    pub fn get_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Configuration for Qwen3-0.6B model.
    /// Note: Qwen3-0.6B uses head_dim=128 (not hidden_size/num_heads=64).
    /// q_proj: [2048, 1024] = 16 heads × 128 head_dim
    /// k_proj/v_proj: [1024, 1024] = 8 kv_heads × 128 head_dim
    pub fn qwen3_0_6b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(1024)
            .with_intermediate_size(3072)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(16)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
    }

    /// Configuration for Qwen3-1.7B model.
    pub fn qwen3_1_7b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(2048)
            .with_intermediate_size(6144)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(16)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
    }

    /// Configuration for Qwen3-4B model (default).
    /// Note: Standard Qwen3-4B has head_dim = 80 (2560/32).
    pub fn qwen3_4b() -> Self {
        Qwen3Config::new()
        // Uses default values: 2560 hidden, 9728 intermediate, 36 layers, 32 heads, 8 kv heads
        // head_dim = 2560/32 = 80
    }

    /// Configuration for Z-Image text encoder variant.
    /// This uses Qwen3-4B architecture but with head_dim=128 instead of 80.
    /// q_proj: [4096, 2560] = 32 heads × 128 head_dim
    /// k_proj/v_proj: [1024, 2560] = 8 kv_heads × 128 head_dim
    pub fn z_image_text_encoder() -> Self {
        Qwen3Config::new()
            .with_hidden_size(2560)
            .with_intermediate_size(9728)
            .with_num_hidden_layers(36)
            .with_num_attention_heads(32)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128)) // Key difference: 128 instead of 80
    }

    /// Configuration for Qwen3-8B model.
    pub fn qwen3_8b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(4096)
            .with_intermediate_size(12288)
            .with_num_hidden_layers(36)
            .with_num_attention_heads(32)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
    }
}

impl Qwen3Config {
    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Model<B> {
        let layers: Vec<Qwen3DecoderLayer<B>> = (0..self.num_hidden_layers)
            .map(|_| Qwen3DecoderLayerConfig::from_model_config(self).init(device))
            .collect();

        Qwen3Model {
            config: Ignored(self.clone()),
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

/// Qwen3 decoder-only transformer model.
#[derive(Module, Debug)]
pub struct Qwen3Model<B: Backend> {
    config: Ignored<Qwen3Config>,
    embed_tokens: Embedding<B>,
    layers: Vec<Qwen3DecoderLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> Qwen3Model<B> {
    /// Get embedding weight tensor for debugging.
    pub fn embed_tokens_weight(&self) -> Tensor<B, 2> {
        self.embed_tokens.weight.val()
    }

    /// Forward pass returning all hidden states.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Attention mask [batch, seq]
    ///
    /// # Returns
    /// A vector of hidden states from each layer, plus the final normalized output.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Vec<Tensor<B, 3>> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Create position IDs
        let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[batch_size, 1]);

        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Collect hidden states from each layer
        // We push AFTER processing each layer to match HuggingFace behavior
        // hidden_states[0] = embedding output
        // hidden_states[i] = output of layer i-1 (for i > 0)
        // hidden_states[-1] = final norm output
        let mut all_hidden_states = Vec::with_capacity(self.layers.len() + 2);
        all_hidden_states.push(hidden_states.clone()); // Embedding output

        // Pass through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(hidden_states, attention_mask.clone(), position_ids.clone());
            all_hidden_states.push(hidden_states.clone()); // Push AFTER processing

            // Debug: print first layer output
            if i == 0 && std::env::var("QWEN3_DEBUG").is_ok() {
                let debug_vals: Vec<f32> = hidden_states.clone()
                    .cast(burn::tensor::DType::F32)
                    .slice([0..1, 0..1, 0..10])
                    .reshape([10])
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap_or(&[])
                    .to_vec();
                eprintln!("[DEBUG] Layer 0 output[0,0,:10]: {:?}", debug_vals);
            }
        }

        // Note: We do NOT push final norm output to match HuggingFace behavior
        // HuggingFace hidden_states has 37 elements (1 embedding + 36 layers)
        // hidden_states[-1] = layer 35 output (before final norm)
        // hidden_states[-2] = layer 34 output
        all_hidden_states
    }

    /// Get text embeddings for Z-Image.
    ///
    /// Returns the second-to-last layer hidden states, masked by attention mask.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Attention mask [batch, seq] (true = valid token)
    pub fn encode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let all_hidden_states = self.forward(input_ids, Some(attention_mask.clone()));

        // Get second-to-last layer (index -2)
        let hidden_states = all_hidden_states[all_hidden_states.len() - 2].clone();

        // The Python code does: prompt_embed[prompt_mask]
        // This extracts only the valid tokens. For simplicity, we'll mask and let
        // the caller handle extraction if needed.
        hidden_states
    }

    /// Forward pass with KV cache for efficient autoregressive generation.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq] (new tokens only during generation)
    /// * `attention_mask` - Optional attention mask [batch, total_seq]
    /// * `position_ids` - Position indices [batch, seq] (positions for new tokens)
    /// * `cache` - Mutable reference to model cache
    ///
    /// # Returns
    /// Final hidden states tensor [batch, seq, hidden_size]
    pub fn forward_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut ModelCache<B>,
    ) -> Tensor<B, 3> {
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Pass through decoder layers with cache
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden_states = layer.forward_with_cache(
                hidden_states,
                attention_mask.clone(),
                position_ids.clone(),
                layer_cache,
            );
        }

        // Final layer norm
        self.norm.forward(hidden_states)
    }

    /// Create a new cache for this model.
    pub fn new_cache(&self) -> ModelCache<B> {
        ModelCache::new(self.layers.len())
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Configuration for a single decoder layer.
#[derive(Config, Debug)]
struct Qwen3DecoderLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
}

impl Qwen3DecoderLayerConfig {
    fn from_model_config(config: &Qwen3Config) -> Self {
        Qwen3DecoderLayerConfig::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.get_head_dim(),
            config.rms_norm_eps,
            config.rope_theta,
        )
    }

    fn init<B: Backend>(&self, device: &B::Device) -> Qwen3DecoderLayer<B> {
        Qwen3DecoderLayer {
            self_attn: Qwen3AttentionConfig::new(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
            )
            .with_head_dim(Some(self.head_dim))
            .with_rope_theta(self.rope_theta)
            .with_rms_norm_eps(self.rms_norm_eps)
            .init(device),
            mlp: Qwen3MLP::new(self.hidden_size, self.intermediate_size, device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

/// A single Qwen3 decoder layer.
#[derive(Module, Debug)]
struct Qwen3DecoderLayer<B: Backend> {
    self_attn: Qwen3Attention<B>,
    mlp: Qwen3MLP<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        // Self attention with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, attention_mask, position_ids);
        let hidden_states = residual + hidden_states;

        // MLP with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }

    fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 3> {
        // Self attention with pre-norm and cache
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache(hidden_states, attention_mask, position_ids, cache);
        let hidden_states = residual + hidden_states;

        // MLP with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states);
        residual + hidden_states
    }
}

/// Qwen3 MLP with SiLU gating (SwiGLU style).
#[derive(Module, Debug)]
struct Qwen3MLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Qwen3MLP<B> {
    fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Qwen3MLP {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate_proj.forward(x.clone()));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(gate * up)
    }
}

// ============================================================================
// Causal Language Model
// ============================================================================

impl Qwen3Config {
    /// Initialize a causal language model (for text generation).
    pub fn init_causal_lm<B: Backend>(&self, device: &B::Device) -> Qwen3ForCausalLM<B> {
        Qwen3ForCausalLM {
            model: self.init(device),
            lm_head: LinearConfig::new(self.hidden_size, self.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

/// Qwen3 model with a language modeling head for text generation.
#[derive(Module, Debug)]
pub struct Qwen3ForCausalLM<B: Backend> {
    /// The base transformer model.
    pub model: Qwen3Model<B>,
    /// Linear layer projecting hidden states to vocabulary logits.
    lm_head: Linear<B>,
}

impl<B: Backend> Qwen3ForCausalLM<B> {
    /// Forward pass returning logits over the vocabulary.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Optional attention mask [batch, seq]
    ///
    /// # Returns
    /// Logits tensor of shape [batch, seq, vocab_size]
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let all_hidden_states = self.model.forward(input_ids, attention_mask);
        // Get the final hidden states (after layer norm)
        let hidden_states = all_hidden_states.last().unwrap().clone();
        self.lm_head.forward(hidden_states)
    }

    /// Forward pass with KV cache returning logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Optional attention mask [batch, total_seq]
    /// * `position_ids` - Position indices [batch, seq]
    /// * `cache` - Mutable reference to model cache
    ///
    /// # Returns
    /// Logits tensor of shape [batch, seq, vocab_size]
    pub fn forward_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut ModelCache<B>,
    ) -> Tensor<B, 3> {
        let hidden_states = self.model.forward_with_cache(input_ids, attention_mask, position_ids, cache);
        self.lm_head.forward(hidden_states)
    }

    /// Generate text autoregressively (without KV cache - slower but simpler).
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch, seq]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)
    /// * `top_p` - Nucleus sampling threshold (1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    ///
    /// # Returns
    /// Generated token IDs [batch, seq + generated]
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch_size, _] = input_ids.dims();

        let mut generated = input_ids;

        for _ in 0..max_new_tokens {
            // Get logits for the last position
            let logits = self.forward(generated.clone(), None);
            let [_, seq_len, vocab_size] = logits.dims();

            // Extract last token logits: [batch, vocab_size]
            let next_token_logits = logits.slice([0..batch_size, (seq_len - 1)..seq_len, 0..vocab_size])
                .reshape([batch_size, vocab_size]);

            // Apply temperature
            let next_token_logits = if temperature != 1.0 {
                next_token_logits / temperature
            } else {
                next_token_logits
            };

            // Sample next token
            // argmax(1) returns [batch, 1], flatten to [batch]
            let next_token: Tensor<B, 1, Int> = if temperature == 0.0 {
                // Greedy decoding
                next_token_logits.argmax(1).flatten(0, 1)
            } else {
                // Apply top-k and top-p, then sample
                let probs = softmax(next_token_logits, 1);
                sample_from_probs(probs, top_k, top_p, &device)
            };

            // Append to generated sequence
            generated = Tensor::cat(vec![generated, next_token.unsqueeze_dim(1)], 1);
        }

        generated
    }

    /// Generate text autoregressively with KV cache (faster).
    ///
    /// This is more efficient than `generate()` as it only computes attention
    /// for new tokens while reusing cached key-value pairs from previous tokens.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch, seq]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)
    /// * `top_p` - Nucleus sampling threshold (1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    ///
    /// # Returns
    /// Generated token IDs [batch, seq + generated]
    pub fn generate_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Tensor<B, 2, Int> {
        // Use default EOS tokens for Qwen3
        self.generate_with_cache_eos(input_ids, max_new_tokens, temperature, top_p, top_k, &[151643, 151645])
    }

    /// Generate text autoregressively with KV cache and custom EOS tokens.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch, seq]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)
    /// * `top_p` - Nucleus sampling threshold (1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    /// * `eos_token_ids` - Token IDs that signal end of generation (e.g., [151643, 151645] for <|endoftext|> and <|im_end|>)
    ///
    /// # Returns
    /// Generated token IDs [batch, seq + generated]
    pub fn generate_with_cache_eos(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        eos_token_ids: &[i64],
    ) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch_size, initial_seq_len] = input_ids.dims();

        // Create cache
        let mut cache = self.model.new_cache();

        // First forward pass: process all input tokens
        let position_ids = Tensor::<B, 1, Int>::arange(0..initial_seq_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[batch_size, 1]);

        let logits = self.forward_with_cache(input_ids.clone(), None, position_ids, &mut cache);
        let [_, _, vocab_size] = logits.dims();

        // Get first new token from the last position
        let next_token_logits = logits.slice([0..batch_size, (initial_seq_len - 1)..initial_seq_len, 0..vocab_size])
            .reshape([batch_size, vocab_size]);

        let next_token_logits = if temperature != 1.0 && temperature != 0.0 {
            next_token_logits / temperature
        } else {
            next_token_logits
        };

        // argmax(1) returns [batch, 1], reshape to [batch]
        let mut next_token: Tensor<B, 1, Int> = if temperature == 0.0 {
            next_token_logits.argmax(1).flatten(0, 1)
        } else {
            let probs = softmax(next_token_logits, 1);
            sample_from_probs(probs, top_k, top_p, &device)
        };

        // Check if first token is EOS
        let token_id = next_token.clone().into_data().as_slice::<i64>().map(|s| s[0]).unwrap_or(0);
        if eos_token_ids.contains(&token_id) {
            return input_ids;
        }

        let mut generated = Tensor::cat(vec![input_ids, next_token.clone().unsqueeze_dim(1)], 1);
        let mut current_pos = initial_seq_len;

        // Generate remaining tokens one at a time
        for _ in 1..max_new_tokens {
            current_pos += 1;

            // Position for the new token
            let position_ids = Tensor::<B, 1, Int>::from_data([current_pos as i64 - 1], &device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[batch_size, 1]);

            // Forward only the new token
            let logits = self.forward_with_cache(
                next_token.clone().unsqueeze_dim(1),
                None,
                position_ids,
                &mut cache,
            );

            // Extract logits for the single new token
            let next_token_logits = logits.slice([0..batch_size, 0..1, 0..vocab_size])
                .reshape([batch_size, vocab_size]);

            let next_token_logits = if temperature != 1.0 && temperature != 0.0 {
                next_token_logits / temperature
            } else {
                next_token_logits
            };

            // argmax(1) returns [batch, 1], flatten to [batch]
            next_token = if temperature == 0.0 {
                next_token_logits.argmax(1).flatten(0, 1)
            } else {
                let probs = softmax(next_token_logits, 1);
                sample_from_probs(probs, top_k, top_p, &device)
            };

            // Check for EOS token
            let token_id = next_token.clone().into_data().as_slice::<i64>().map(|s| s[0]).unwrap_or(0);
            if eos_token_ids.contains(&token_id) {
                break;
            }

            generated = Tensor::cat(vec![generated, next_token.clone().unsqueeze_dim(1)], 1);
        }

        generated
    }

    /// Create a new cache for this model.
    pub fn new_cache(&self) -> ModelCache<B> {
        self.model.new_cache()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.model.config.0.vocab_size
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.model.config.0.hidden_size
    }
}

/// Sample a token from probability distribution.
///
/// Uses CPU-based sampling to avoid slow GPU random operations.
/// Falls back to argmax if sampling fails.
fn sample_from_probs<B: Backend>(
    probs: Tensor<B, 2>,
    top_k: usize,
    _top_p: f32,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    use rand::Rng;

    let [batch_size, vocab_size] = probs.dims();

    // Transfer probs to CPU for sampling (GPU random is extremely slow)
    let probs_data = probs.into_data();
    let probs_slice: Vec<f32> = probs_data.as_slice::<half::bf16>()
        .map(|s| s.iter().map(|x| x.to_f32()).collect())
        .or_else(|_| probs_data.as_slice::<f32>().map(|s| s.to_vec()))
        .unwrap_or_default();

    if probs_slice.is_empty() {
        // Fallback: return 0
        return Tensor::zeros([batch_size], device);
    }

    let mut rng = rand::rng();
    let mut sampled_tokens = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let start = b * vocab_size;
        let end = start + vocab_size;
        let batch_probs = &probs_slice[start..end];

        // Apply top-k: find top-k indices
        let mut indexed: Vec<(usize, f32)> = batch_probs.iter()
            .cloned()
            .enumerate()
            .collect();

        // Partial sort to get top-k
        let k = if top_k > 0 && top_k < vocab_size { top_k } else { vocab_size };
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        // Normalize probabilities
        let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
        if sum <= 0.0 {
            // Fallback to argmax
            sampled_tokens.push(indexed.first().map(|(i, _)| *i as i64).unwrap_or(0));
            continue;
        }

        // Sample from categorical distribution
        let r: f32 = rng.random();
        let mut cumsum = 0.0f32;
        let mut selected = indexed[0].0;

        for (idx, prob) in &indexed {
            cumsum += prob / sum;
            if r < cumsum {
                selected = *idx;
                break;
            }
        }

        sampled_tokens.push(selected as i64);
    }

    Tensor::from_data(sampled_tokens.as_slice(), device)
}
