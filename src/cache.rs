//! KV Cache for efficient autoregressive generation.
//!
//! During autoregressive generation, we only need to compute attention for the
//! new token while reusing the cached key-value pairs from previous tokens.

use burn::{prelude::Backend, tensor::Tensor};

/// KV cache for a single attention layer.
///
/// Stores the key and value tensors from previous forward passes,
/// allowing efficient incremental decoding.
#[derive(Debug, Clone)]
pub struct KVCache<B: Backend> {
    /// Cached keys: [batch, seq_len, num_kv_heads, head_dim]
    pub key: Option<Tensor<B, 4>>,
    /// Cached values: [batch, seq_len, num_kv_heads, head_dim]
    pub value: Option<Tensor<B, 4>>,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    /// Create a new empty cache.
    pub fn new() -> Self {
        KVCache {
            key: None,
            value: None,
        }
    }

    /// Update the cache with new key-value pairs.
    ///
    /// Concatenates the new K/V with existing cached values along the sequence dimension.
    ///
    /// # Arguments
    /// * `new_key` - New key tensor [batch, new_seq_len, num_kv_heads, head_dim]
    /// * `new_value` - New value tensor [batch, new_seq_len, num_kv_heads, head_dim]
    ///
    /// # Returns
    /// The full key and value tensors including cached values.
    pub fn update(
        &mut self,
        new_key: Tensor<B, 4>,
        new_value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let (key, value) = match (&self.key, &self.value) {
            (Some(cached_key), Some(cached_value)) => {
                // Concatenate along sequence dimension (dim 1)
                let key = Tensor::cat(vec![cached_key.clone(), new_key], 1);
                let value = Tensor::cat(vec![cached_value.clone(), new_value], 1);
                (key, value)
            }
            _ => {
                // First call, no cache yet
                (new_key, new_value)
            }
        };

        // Store for next iteration
        self.key = Some(key.clone());
        self.value = Some(value.clone());

        (key, value)
    }

    /// Get the current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        self.key.as_ref().map(|k| k.dims()[1]).unwrap_or(0)
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
    }
}

/// Cache for all layers in the model.
#[derive(Debug)]
pub struct ModelCache<B: Backend> {
    /// Per-layer KV caches.
    pub layers: Vec<KVCache<B>>,
}

impl<B: Backend> ModelCache<B> {
    /// Create a new cache for a model with the given number of layers.
    pub fn new(num_layers: usize) -> Self {
        ModelCache {
            layers: (0..num_layers).map(|_| KVCache::new()).collect(),
        }
    }

    /// Get the current sequence length (from the first layer's cache).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for cache in &mut self.layers {
            cache.clear();
        }
    }
}
