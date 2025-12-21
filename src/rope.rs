//! Rotary Position Embeddings for Qwen3.
//!
//! Qwen3 uses RoPE with theta=1000000 and applies it differently than Z-Image's
//! image transformer.

use burn::{Tensor, prelude::Backend, tensor::Int};

/// Compute rotary position embeddings for Qwen3.
///
/// # Arguments
/// * `positions` - Position indices [batch_size, seq_len]
/// * `head_dim` - Dimension per attention head
/// * `theta` - RoPE theta parameter (default 1000000 for Qwen3)
/// * `device` - Device to create tensors on
pub fn compute_rope_embeddings<B: Backend>(
    positions: Tensor<B, 2, Int>,
    head_dim: usize,
    theta: f64,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let half_dim = head_dim / 2;

    // Compute frequency bands: 1 / (theta^(2i/d)) for i in 0..half_dim
    let freq_seq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let freqs = Tensor::<B, 1>::from_floats(freq_seq.as_slice(), device);

    // positions: [batch, seq] -> [batch, seq, 1]
    // freqs: [half_dim] -> [1, 1, half_dim]
    // angles: [batch, seq, half_dim]
    let positions_float = positions.float();
    let angles = positions_float.unsqueeze_dim::<3>(2) * freqs.unsqueeze::<3>();

    // cos and sin: [batch, seq, half_dim] -> [batch, seq, 1, half_dim]
    let cos = angles.clone().cos().unsqueeze_dim::<4>(2);
    let sin = angles.sin().unsqueeze_dim::<4>(2);

    (cos, sin)
}

/// Apply rotary embeddings to query and key tensors.
///
/// # Arguments
/// * `q` - Query tensor [batch, seq, n_heads, head_dim]
/// * `k` - Key tensor [batch, seq, n_kv_heads, head_dim]
/// * `cos` - Cosine embeddings [batch, seq, 1, half_dim]
/// * `sin` - Sine embeddings [batch, seq, 1, half_dim]
pub fn apply_rope<B: Backend>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let q_rotated = rotate_half(q, cos.clone(), sin.clone());
    let k_rotated = rotate_half(k, cos, sin);
    (q_rotated, k_rotated)
}

/// Rotate half of the tensor dimensions using RoPE.
fn rotate_half<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, seq, n_heads, head_dim] = x.dims();
    let half_dim = head_dim / 2;

    // Split into first and second halves
    let x1 = x.clone().slice([0..batch, 0..seq, 0..n_heads, 0..half_dim]);
    let x2 = x.slice([0..batch, 0..seq, 0..n_heads, half_dim..head_dim]);

    // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    let rotated_x1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let rotated_x2 = x1 * sin + x2 * cos;

    // Concatenate back
    Tensor::cat(vec![rotated_x1, rotated_x2], 3)
}
