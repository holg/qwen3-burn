//! Weight loading utilities for Qwen3 models.

use std::path::PathBuf;

use burn::{
    prelude::Backend,
    store::{BurnpackStore, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore},
};
use rootcause::{Report, prelude::ResultExt};
use thiserror::Error;

use crate::{Qwen3ForCausalLM, Qwen3Model};

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("Error while loading weights")]
    LoadError,
    #[error("Unrecognised file extension")]
    UnknownExtension,
}

/// Create a SafetensorsStore with HuggingFace-to-Burn key remapping for Qwen3Model.
/// This removes the "model." prefix since Qwen3Model doesn't have that wrapper.
fn create_safetensors_store_base(path: PathBuf) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        // Remove "model." prefix (for base model weights)
        .with_key_remapping(r"^model\.", "")
        // RmsNorm uses gamma in burn
        .with_key_remapping(r"\.weight$", ".gamma")
        // But Linear layers use weight, not gamma
        .with_key_remapping(r"_proj\.gamma$", "_proj.weight")
        .with_key_remapping(r"embed_tokens\.gamma$", "embed_tokens.weight")
}

/// Create a SafetensorsStore with HuggingFace-to-Burn key remapping for Qwen3ForCausalLM.
/// This keeps the "model." prefix since Qwen3ForCausalLM wraps Qwen3Model in a `model` field.
fn create_safetensors_store_causal_lm(path: PathBuf) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        // RmsNorm uses gamma in burn
        .with_key_remapping(r"\.weight$", ".gamma")
        // But Linear layers use weight, not gamma
        .with_key_remapping(r"_proj\.gamma$", "_proj.weight")
        .with_key_remapping(r"embed_tokens\.gamma$", "embed_tokens.weight")
        .with_key_remapping(r"lm_head\.gamma$", "lm_head.weight")
}

impl<B: Backend> Qwen3Model<B> {
    /// Load weights and return self (builder pattern).
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, Report<ModelLoadError>> {
        self.load_weights(path)?;
        Ok(self)
    }

    /// Load weights from a file.
    ///
    /// Supports `.safetensors` (HuggingFace format) and `.bpk` (Burn format).
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                let mut weights = create_safetensors_store_base(path);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path)
                    .auto_extension(false)
                    .zero_copy(true);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            _ => {
                return Err(Report::new(ModelLoadError::UnknownExtension));
            }
        }

        Ok(())
    }
}

impl<B: Backend> Qwen3ForCausalLM<B> {
    /// Load weights and return self (builder pattern).
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, Report<ModelLoadError>> {
        self.load_weights(path)?;
        Ok(self)
    }

    /// Load weights from a file.
    ///
    /// Supports `.safetensors` (HuggingFace format) and `.bpk` (Burn format).
    /// For HuggingFace format, expects the standard Qwen3ForCausalLM weight structure.
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                let mut weights = create_safetensors_store_causal_lm(path);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path)
                    .auto_extension(false)
                    .zero_copy(true);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            _ => {
                return Err(Report::new(ModelLoadError::UnknownExtension));
            }
        }

        Ok(())
    }
}
