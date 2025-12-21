//! Grouped Query Attention for Qwen3.

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, module::attention, ops::AttentionModuleOptions},
};

use super::cache::KVCache;
use super::rope::{apply_rope, compute_rope_embeddings};

/// Configuration for Qwen3 attention.
#[derive(Config, Debug)]
pub struct Qwen3AttentionConfig {
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,
    /// Explicit head dimension. If None, computed as hidden_size / num_attention_heads.
    pub head_dim: Option<usize>,
    /// RoPE theta parameter.
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
}

impl Qwen3AttentionConfig {
    /// Initialize the attention module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Attention<B> {
        let head_dim = self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads);

        Qwen3Attention {
            num_heads: Ignored(self.num_attention_heads),
            num_kv_heads: Ignored(self.num_key_value_heads),
            head_dim: Ignored(head_dim),
            rope_theta: Ignored(self.rope_theta),
            q_proj: LinearConfig::new(self.hidden_size, self.num_attention_heads * head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, self.num_key_value_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, self.num_key_value_heads * head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(self.num_attention_heads * head_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

/// Grouped Query Attention module for Qwen3.
#[derive(Module, Debug)]
pub struct Qwen3Attention<B: Backend> {
    num_heads: Ignored<usize>,
    num_kv_heads: Ignored<usize>,
    head_dim: Ignored<usize>,
    rope_theta: Ignored<f64>,

    q_proj: Linear<B>,
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    o_proj: Linear<B>,
    q_norm: RmsNorm<B>,
    k_norm: RmsNorm<B>,
}

impl<B: Backend> Qwen3Attention<B> {
    /// Forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq, hidden_size]
    /// * `attention_mask` - Optional attention mask [batch, seq]
    /// * `position_ids` - Position indices [batch, seq]
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, burn::tensor::Int>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();

        // Project to Q, K, V
        let query = self.q_proj.forward(hidden_states.clone());
        let key = self.k_proj.forward(hidden_states.clone());
        let value = self.v_proj.forward(hidden_states);

        // Reshape to [batch, seq, n_heads, head_dim]
        let query = query.reshape([batch_size, seq_len, *self.num_heads, *self.head_dim]);
        let key = key.reshape([batch_size, seq_len, *self.num_kv_heads, *self.head_dim]);
        let value = value.reshape([batch_size, seq_len, *self.num_kv_heads, *self.head_dim]);

        // Apply QK normalization
        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        // Compute and apply RoPE
        let (cos, sin) = compute_rope_embeddings(position_ids, *self.head_dim, *self.rope_theta, &device);
        let (query, key) = apply_rope(query, key, cos, sin);

        // Expand KV heads for GQA
        let n_rep = *self.num_heads / *self.num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        // Attention: transpose to [batch, n_heads, seq, head_dim]
        // Create causal mask: lower triangular matrix where true = attend
        // Use float operations to avoid I64 issues on Metal
        let row_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let col_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

        let rows = Tensor::<B, 1>::from_floats(row_idx.as_slice(), &device)
            .unsqueeze_dim::<2>(1)
            .repeat(&[1, seq_len]);  // [seq, seq]
        let cols = Tensor::<B, 1>::from_floats(col_idx.as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[seq_len, 1]);  // [seq, seq]

        // Causal mask: rows < cols (upper triangular excluding diagonal) = positions to MASK OUT
        // Burn attention uses true = mask out (fill with -inf)
        let causal_mask = rows.lower(cols);  // [seq, seq] Bool, true = future positions to mask

        // Combine with optional attention mask (padding mask)
        let combined_mask = match attention_mask {
            Some(pad_mask) => {
                // pad_mask: [batch, seq] where true = valid token, false = padding
                // We need to mask where pad_mask is false, so invert: true = masked position
                let pad_mask_inverted = pad_mask.bool_not();  // true = padding (mask out)
                // Expand to [batch, 1, seq_q, seq_k] for attention scores
                // For self-attention, seq_q = seq_k = seq
                let pad_expanded = pad_mask_inverted.unsqueeze_dims::<4>(&[1, 2])  // [batch, 1, 1, seq_k]
                    .repeat(&[1, 1, seq_len, 1]);  // [batch, 1, seq_q, seq_k]
                let causal_expanded = causal_mask.unsqueeze_dims::<4>(&[0, 1]);  // [1, 1, seq_q, seq_k]
                // Combined: mask where either is true (padding OR future)
                pad_expanded.bool_or(causal_expanded)
            }
            None => causal_mask.unsqueeze_dims::<4>(&[0, 1]),  // [1, 1, seq, seq]
        };

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            Some(combined_mask),
            None,
            AttentionModuleOptions::default(),
        );

        // Reshape back to [batch, seq, hidden_size]
        let attn_output = attn_output.movedim(1, 2).reshape([
            batch_size as i64,
            seq_len as i64,
            (*self.num_heads * *self.head_dim) as i64,
        ]);

        self.o_proj.forward(attn_output)
    }

    /// Forward pass with KV cache for efficient autoregressive generation.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq, hidden_size] (usually seq=1 for generation)
    /// * `attention_mask` - Optional attention mask [batch, total_seq] (including cached positions)
    /// * `position_ids` - Position indices [batch, seq] (positions for new tokens only)
    /// * `cache` - Mutable reference to KV cache for this layer
    ///
    /// # Returns
    /// Output tensor [batch, seq, hidden_size]
    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, burn::tensor::Int>,
        cache: &mut KVCache<B>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();

        // Project to Q, K, V for new tokens only
        let query = self.q_proj.forward(hidden_states.clone());
        let key = self.k_proj.forward(hidden_states.clone());
        let value = self.v_proj.forward(hidden_states);

        // Reshape to [batch, seq, n_heads, head_dim]
        let query = query.reshape([batch_size, seq_len, *self.num_heads, *self.head_dim]);
        let key = key.reshape([batch_size, seq_len, *self.num_kv_heads, *self.head_dim]);
        let value = value.reshape([batch_size, seq_len, *self.num_kv_heads, *self.head_dim]);

        // Apply QK normalization
        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        // Compute and apply RoPE (only for new positions)
        let (cos, sin) = compute_rope_embeddings(position_ids, *self.head_dim, *self.rope_theta, &device);
        let (query, key) = apply_rope(query, key, cos, sin);

        // Update cache and get full K, V (including past)
        let (key, value) = cache.update(key, value);

        // Expand KV heads for GQA
        let n_rep = *self.num_heads / *self.num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        // Attention: transpose to [batch, n_heads, seq, head_dim]
        // Query is [batch, new_seq, n_heads, head_dim]
        // Key/Value are [batch, total_seq, n_heads, head_dim]
        //
        // For the prefill phase (seq_len > 1), we need a causal mask to prevent
        // tokens from attending to future positions.
        // For the decode phase (seq_len == 1), no causal mask is needed.
        let [_, total_seq, _, _] = key.dims();

        let combined_mask = if seq_len > 1 {
            // Prefill: create causal mask [seq_len, total_seq]
            // Causal mask: rows < cols = positions to MASK OUT (future tokens)
            let row_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
            let col_idx: Vec<f32> = (0..total_seq).map(|i| i as f32).collect();

            let rows = Tensor::<B, 1>::from_floats(row_idx.as_slice(), &device)
                .unsqueeze_dim::<2>(1)
                .repeat(&[1, total_seq]);  // [seq_len, total_seq]
            let cols = Tensor::<B, 1>::from_floats(col_idx.as_slice(), &device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[seq_len, 1]);  // [seq_len, total_seq]

            // Causal mask: rows < cols = future positions to mask
            let causal_mask = rows.lower(cols);  // [seq_len, total_seq] Bool

            // Combine with optional attention mask (padding mask)
            match attention_mask {
                Some(pad_mask) => {
                    let pad_mask_inverted = pad_mask.bool_not();
                    let pad_expanded = pad_mask_inverted.unsqueeze_dims::<4>(&[1, 2])
                        .repeat(&[1, 1, seq_len, 1]);
                    let causal_expanded = causal_mask.unsqueeze_dims::<4>(&[0, 1]);
                    Some(pad_expanded.bool_or(causal_expanded))
                }
                None => Some(causal_mask.unsqueeze_dims::<4>(&[0, 1])),
            }
        } else {
            // Decode phase: no causal mask needed (query is single token)
            attention_mask.map(|m| m.unsqueeze_dims(&[1, 2]))
        };

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            combined_mask,
            None,
            AttentionModuleOptions::default(),
        );

        // Reshape back to [batch, seq, hidden_size]
        let attn_output = attn_output.movedim(1, 2).reshape([
            batch_size as i64,
            seq_len as i64,
            (*self.num_heads * *self.head_dim) as i64,
        ]);

        self.o_proj.forward(attn_output)
    }
}
