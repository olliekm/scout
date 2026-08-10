//! Model loading -- architecture component 1 from AGENT.md: reads
//! safetensors weights off disk into memory. This module deliberately
//! stops at "here are the named tensors, as owned CPU-side bytes" --
//! organizing them into a real per-layer Qwen2 model graph (embedding, N
//! transformer layers, final norm, lm_head) is later scope, once this
//! flat loading step is solid. Same "smallest independently-testable
//! slice first" pattern as block_allocator.rs -> scheduler.rs.
//!
//! Uses the `safetensors` crate for the file format itself (JSON header +
//! raw tensor byte offsets) -- same "integrate, don't reinvent" call as
//! FlashAttention-2 (step 5): parsing a fixed binary format isn't where
//! this project's systems-engineering story lives.
//!
//! New Rust idea versus block_allocator.rs/scheduler.rs: this is the
//! first place in the engine with a REAL custom error type instead of
//! `Option`. `Option` works when there's exactly one way to fail and the
//! caller doesn't need to know why (e.g. "pool exhausted"). Loading a
//! file has (at least) two structurally different failure causes -- the
//! file might not exist / not be readable (an `io::Error`), or it might
//! exist but not be valid safetensors (a `safetensors::SafeTensorError`)
//! -- and collapsing both into a bare `None` would throw away exactly the
//! information a caller needs to report a useful error. `Result<T, E>`
//! with a custom `E` is Rust's tool for "might fail, in one of several
//! specific ways, and the caller needs to know which."

use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// One tensor's data, fully OWNED (not borrowed from the file buffer it
/// came from). This matters because of a lifetime constraint you'll hit
/// directly: `SafeTensors::deserialize` borrows from whatever byte buffer
/// you hand it, and every `TensorView` it returns borrows from THAT --
/// so a `TensorView`'s data cannot outlive the buffer `load_safetensors`
/// reads the file into. Since this type needs to be returned OUT of
/// `load_safetensors` and outlive that local buffer, it has to copy the
/// bytes out into its own `Vec<u8>` rather than hold a `TensorView`
/// directly (same "OWN it, don't borrow it" idea as `PagedGpuBuffer`
/// owning a `GpuBuffer` instead of a raw pointer).
pub struct Tensor {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub data: Vec<u8>,
}

/// Everything that can go wrong loading a safetensors file, kept as
/// separate variants (not collapsed into one) because the two causes are
/// genuinely different and a caller might want to handle them
/// differently -- e.g. "retry with a different path" for `Io` vs. "this
/// file is corrupt/wrong format" for `Parse`.
#[derive(Debug)]
pub enum ModelLoadError {
    Io(std::io::Error),
    Parse(safetensors::SafeTensorError),

    /// The flat map from `load_safetensors` was missing a tensor the
    /// Qwen2 graph expects by name (e.g. a typo'd key, or the file being
    /// a different model/architecture than expected). Distinct from
    /// `Parse` -- the FILE parsed fine, it just didn't contain what this
    /// specific model graph needs. Carries the expected name so the error
    /// message can say exactly what's missing, not just "something is."
    MissingTensor(String),
}

// `?` on a `Result<_, io::Error>` or `Result<_, SafeTensorError>` inside a
// function returning `Result<_, ModelLoadError>` only compiles if Rust
// knows how to convert those error types INTO `ModelLoadError` -- that's
// what `From` is for. Each impl is a one-line wrap into the matching
// variant.
impl From<std::io::Error> for ModelLoadError {
    fn from(err: std::io::Error) -> Self {
        ModelLoadError::Io(err)
    }
}

impl From<safetensors::SafeTensorError> for ModelLoadError {
    fn from(err: safetensors::SafeTensorError) -> Self {
        ModelLoadError::Parse(err)
    }
}

/// Load every tensor out of a single .safetensors file into a flat,
/// name-indexed map -- keys like "model.embed_tokens.weight",
/// "model.layers.0.self_attn.q_proj.weight", exactly as they're named in
/// the file. Organizing these into a structured per-layer model is a
/// later step; this one just gets everything off disk and into owned
/// memory.
///
/// Steps:
///   1. Read the whole file into a `Vec<u8>`: `std::fs::read(path)?` --
///      the `?` here relies on the `From<io::Error>` impl above.
///   2. `SafeTensors::deserialize(&buffer)?` -- same `?`/`From` trick,
///      this time for `SafeTensorError`.
///   3. Build an empty `HashMap<String, Tensor>`.
///   4. For each `(name, view)` in `safetensors.tensors()`: construct a
///      `Tensor` by copying `view.shape().to_vec()`, `view.dtype()`, and
///      `view.data().to_vec()` (the actual byte copy this type's doc
///      comment explains the need for), insert into the map under `name`.
///   5. Return `Ok(map)`.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Tensor>, ModelLoadError> {
    let buffer = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&buffer)?;
    let mut str_tensor: HashMap<String, Tensor> = HashMap::new();
    for (name, view) in tensors.tensors() {
        let tensor = Tensor {
            shape: view.shape().to_vec(), dtype: view.dtype(), data: view.data().to_vec()
        };
        str_tensor.insert(name, tensor);
    }
    Ok(str_tensor)
}

/// Qwen2.5-Coder-7B-Instruct's architecture, per AGENT.md -- hardcoded
/// rather than read from the model's config.json, since this project
/// targets exactly one checkpoint (see the "hardcode it" call: no other
/// model needs to load through this path, so a config-file parser would
/// be speculative generality with nothing real to justify it yet).
///
/// Only `NUM_LAYERS` is actually used by this slice (it's the loop bound
/// for pulling per-layer tensors out of the flat map) -- HIDDEN_SIZE and
/// the head counts aren't consumed by anything yet, so they're left as a
/// comment, not constants, until real compute code needs them:
///   hidden_size: 3584, num_query_heads: 28, num_kv_heads: 4 (GQA),
///   max_context_len: 32768
pub const NUM_LAYERS: usize = 28;

/// One transformer layer's weights, named to match the tensor keys
/// they're pulled from (e.g. `q_proj_weight` <- "...self_attn.q_proj.
/// weight"). Qwen2 (unlike plain Llama) has biases on q/k/v projections,
/// not just weights -- that's why there are `_bias` fields on q/k/v but
/// not on `o_proj` or the MLP projections, which Qwen2 leaves bias-free.
pub struct Qwen2Layer {
    pub input_layernorm_weight: Tensor,
    pub q_proj_weight: Tensor,
    pub q_proj_bias: Tensor,
    pub k_proj_weight: Tensor,
    pub k_proj_bias: Tensor,
    pub v_proj_weight: Tensor,
    pub v_proj_bias: Tensor,
    pub o_proj_weight: Tensor,
    pub post_attention_layernorm_weight: Tensor,
    pub gate_proj_weight: Tensor,
    pub up_proj_weight: Tensor,
    pub down_proj_weight: Tensor,
}

/// The full Qwen2ForCausalLM graph: token embedding, NUM_LAYERS
/// transformer layers, a final norm, and the output projection.
///
/// UNVERIFIED: `lm_head.weight` is assumed to be a separate tensor here.
/// Some smaller Qwen2.5 checkpoints tie `lm_head` to `embed_tokens`
/// (`tie_word_embeddings`) and don't ship `lm_head.weight` as its own key
/// at all -- worth checking the real file's `tensors.names()` (or the
/// model's config.json) once you have it on the pod. If it turns out to
/// be tied, `build_qwen2_model` will fail loudly with a `MissingTensor`
/// error rather than silently loading something wrong, which is the
/// point of doing lookups this way instead of guessing.
pub struct Qwen2Model {
    pub embed_tokens: Tensor,
    pub layers: Vec<Qwen2Layer>,
    pub norm: Tensor,
    pub lm_head: Tensor,
}

/// Remove and return the tensor named `name` from `tensors`, or a
/// `MissingTensor` error if it isn't there.
///
/// New Rust idea: `HashMap::remove` returns `Option<V>` -- the VALUE that
/// was there, not a reference to it. That's what makes this useful here:
/// it transfers OWNERSHIP of the `Tensor` out of the map to the caller,
/// no cloning needed (you're taking each tensor out permanently as you
/// build the structured model, not keeping the flat map around
/// afterward). `Option::ok_or_else(|| ...)` converts that `Option<Tensor>`
/// into a `Result<Tensor, ModelLoadError>` -- `Some(t)` becomes `Ok(t)`,
/// `None` becomes `Err` of whatever the closure returns -- which is what
/// lets you use `?` on the result of this function.
fn take(tensors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor, ModelLoadError> {
    tensors.remove(name).ok_or_else(|| ModelLoadError::MissingTensor(format!("There is no tensor with the name: {name}")))
}

/// Build one `Qwen2Layer` by pulling its 12 tensors out of `tensors` by
/// name, for the layer at index `layer_idx`.
///
/// Steps: for each field, call `take` with the matching HuggingFace key,
/// built with `format!`, e.g.:
///   input_layernorm_weight  <- format!("model.layers.{layer_idx}.input_layernorm.weight")
///   q_proj_weight           <- format!("model.layers.{layer_idx}.self_attn.q_proj.weight")
///   q_proj_bias             <- format!("model.layers.{layer_idx}.self_attn.q_proj.bias")
///   k_proj_weight/k_proj_bias, v_proj_weight/v_proj_bias -- same pattern, k_proj/v_proj
///   o_proj_weight           <- format!("model.layers.{layer_idx}.self_attn.o_proj.weight")
///   post_attention_layernorm_weight <- format!("model.layers.{layer_idx}.post_attention_layernorm.weight")
///   gate_proj_weight/up_proj_weight/down_proj_weight <- format!("model.layers.{layer_idx}.mlp.{gate,up,down}_proj.weight")
/// Each `take(...)?` call either gives you the `Tensor` or propagates the
/// `MissingTensor` error straight out of this function via `?`.
fn load_layer(tensors: &mut HashMap<String, Tensor>, layer_idx: usize) -> Result<Qwen2Layer, ModelLoadError> {
    let prefix = format!("model.layers.{layer_idx}");
    Ok(Qwen2Layer {
        input_layernorm_weight: take(tensors, &format!("{prefix}.input_layernorm.weight"))?,
        q_proj_weight: take(tensors, &format!("{prefix}.self_attn.q_proj.weight"))?,
        q_proj_bias: take(tensors, &format!("{prefix}.self_attn.q_proj.bias"))?,
        k_proj_weight: take(tensors, &format!("{prefix}.self_attn.k_proj.weight"))?,
        k_proj_bias: take(tensors, &format!("{prefix}.self_attn.k_proj.bias"))?,
        v_proj_weight: take(tensors, &format!("{prefix}.self_attn.v_proj.weight"))?,
        v_proj_bias: take(tensors, &format!("{prefix}.self_attn.v_proj.bias"))?,
        o_proj_weight: take(tensors, &format!("{prefix}.self_attn.o_proj.weight"))?,
        post_attention_layernorm_weight: take(tensors, &format!("{prefix}.post_attention_layernorm.weight"))?,
        gate_proj_weight: take(tensors, &format!("{prefix}.mlp.gate_proj.weight"))?,
        up_proj_weight: take(tensors, &format!("{prefix}.mlp.up_proj.weight"))?,
        down_proj_weight: take(tensors, &format!("{prefix}.mlp.down_proj.weight"))?,
    })
}

/// Organize a flat, name-indexed tensor map (from `load_safetensors`)
/// into a structured `Qwen2Model`. Takes ownership of `tensors` (not
/// `&HashMap`) since `take`/`load_layer` consume entries out of it as
/// they go -- there's nothing meaningful left in the map afterward, so
/// there's no reason to keep it borrowed and force the caller to manage
/// its lifetime.
///
/// Steps:
///   1. `take(&mut tensors, "model.embed_tokens.weight")?`
///   2. Loop `layer_idx` in `0..NUM_LAYERS`, calling `load_layer(&mut
///      tensors, layer_idx)?` each time, collecting into a `Vec<Qwen2Layer>`
///   3. `take(&mut tensors, "model.norm.weight")?`
///   4. `take(&mut tensors, "lm_head.weight")?` -- see `Qwen2Model`'s doc
///      comment about this one being unverified against the real file
///   5. Construct and return `Ok(Qwen2Model { ... })`
pub fn build_qwen2_model(mut tensors: HashMap<String, Tensor>) -> Result<Qwen2Model, ModelLoadError> {
    let embed_tokens = take(&mut tensors, "model.embed_tokens.weight")?;
    let mut layers: Vec<Qwen2Layer> = Vec::new();
    for layer_idx in 0..NUM_LAYERS {
        layers.push(load_layer(&mut tensors, layer_idx)?);
    }
    let norm = take(&mut tensors, "model.norm.weight")?;
    let lm_head = take(&mut tensors, "lm_head.weight")?;
    
    Ok(Qwen2Model { embed_tokens, layers, norm, lm_head })

}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    #[test]
    fn missing_file_returns_io_error() {
        let result = load_safetensors(Path::new("/nonexistent/does_not_exist.safetensors"));
        assert!(matches!(result, Err(ModelLoadError::Io(_))));
    }

    #[test]
    fn loads_tensors_from_a_real_safetensors_file() {
        // Build a real fixture with safetensors' own serialize_to_file,
        // rather than a committed binary file -- self-contained, and it
        // exercises load_safetensors against the actual file format
        // instead of a guessed-at shape.
        let data: Vec<u8> = vec![0, 0, 128, 63, 0, 0, 0, 64]; // f32 1.0, 2.0 (little-endian)
        let view = TensorView::new(Dtype::F32, vec![2], &data).unwrap();

        let path = std::env::temp_dir().join(format!(
            "scout_model_test_{}_loads_tensors.safetensors",
            std::process::id()
        ));
        safetensors::serialize_to_file(vec![("weight".to_string(), view)], None, &path).unwrap();

        let loaded = load_safetensors(&path).unwrap();
        std::fs::remove_file(&path).ok(); // clean up before asserting, so failures don't leak the file

        assert_eq!(loaded.len(), 1);
        let tensor = &loaded["weight"];
        assert_eq!(tensor.shape, vec![2]);
        assert_eq!(tensor.dtype, Dtype::F32);
        assert_eq!(tensor.data, data);
    }

    /// Every tensor name `build_qwen2_model` is expected to look up, for
    /// all NUM_LAYERS layers plus the three top-level ones -- mirrors the
    /// exact naming pattern in `load_layer`/`build_qwen2_model`, so a
    /// fixture built from this list matches what they actually ask for.
    fn all_expected_qwen2_keys() -> Vec<String> {
        let mut keys = vec!["model.embed_tokens.weight".to_string()];
        for layer_idx in 0..NUM_LAYERS {
            let prefix = format!("model.layers.{layer_idx}");
            keys.push(format!("{prefix}.input_layernorm.weight"));
            keys.push(format!("{prefix}.self_attn.q_proj.weight"));
            keys.push(format!("{prefix}.self_attn.q_proj.bias"));
            keys.push(format!("{prefix}.self_attn.k_proj.weight"));
            keys.push(format!("{prefix}.self_attn.k_proj.bias"));
            keys.push(format!("{prefix}.self_attn.v_proj.weight"));
            keys.push(format!("{prefix}.self_attn.v_proj.bias"));
            keys.push(format!("{prefix}.self_attn.o_proj.weight"));
            keys.push(format!("{prefix}.post_attention_layernorm.weight"));
            keys.push(format!("{prefix}.mlp.gate_proj.weight"));
            keys.push(format!("{prefix}.mlp.up_proj.weight"));
            keys.push(format!("{prefix}.mlp.down_proj.weight"));
        }
        keys.push("model.norm.weight".to_string());
        keys.push("lm_head.weight".to_string());
        keys
    }

    fn tiny_tensor() -> Tensor {
        Tensor { shape: vec![1], dtype: Dtype::F32, data: vec![0, 0, 0, 0] }
    }

    #[test]
    fn build_qwen2_model_fails_when_tensors_missing() {
        // Empty map -- not even embed_tokens is present, so this should
        // fail on the very first take() call, not panic or silently
        // build a broken model.
        let result = build_qwen2_model(HashMap::new());
        assert!(matches!(result, Err(ModelLoadError::MissingTensor(_))));
    }

    #[test]
    fn build_qwen2_model_fails_when_one_layer_tensor_is_missing() {
        // All keys present EXCEPT one deep in the middle (layer 5's
        // k_proj bias) -- confirms the failure isn't just "empty map",
        // it's a real per-tensor check.
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for key in all_expected_qwen2_keys() {
            if key == "model.layers.5.self_attn.k_proj.bias" {
                continue;
            }
            tensors.insert(key, tiny_tensor());
        }
        let result = build_qwen2_model(tensors);
        assert!(matches!(result, Err(ModelLoadError::MissingTensor(_))));
    }

    #[test]
    fn build_qwen2_model_succeeds_with_all_expected_tensors() {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for key in all_expected_qwen2_keys() {
            tensors.insert(key, tiny_tensor());
        }

        let model = build_qwen2_model(tensors).unwrap();
        assert_eq!(model.layers.len(), NUM_LAYERS);
    }
}
