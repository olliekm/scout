// Runs automatically before `cargo build` compiles this crate. Its job:
// invoke nvcc (NVIDIA's CUDA compiler) on gpu_alloc.cu and tell rustc's
// linker where to find the resulting object code, so the `unsafe extern
// "C"` declarations in lib.rs can actually resolve to real machine code.
//
// This requires a CUDA toolchain (nvcc) to be installed and on PATH -- on
// your Mac, there's no NVIDIA GPU or CUDA toolkit, so this crate can ONLY
// be built on the pod. `cargo build` for the workspace as a whole will fail
// here if run locally -- that's expected, not a bug (see engine/, which
// stays pure-Rust and Mac-buildable specifically so allocator/scheduler
// work doesn't require pod access).

use std::path::PathBuf;
use std::process::Command;

// The exact CUTLASS commit flash-attention's own csrc/cutlass git submodule
// is pinned to (checked directly against Dao-AILab/flash-attention's
// .gitmodules + submodule ref on 2026-08-11) -- keeps our vendored
// flash_attn/ headers and CUTLASS in sync with what they were actually
// built/tested against upstream, not just "whatever's newest."
const CUTLASS_COMMIT: &str = "7127592069c2fe01b041e174ba4345ef9b279671";

fn main() {
    cc::Build::new()
        .cuda(true)
        .file("src/gpu_alloc.cu")
        .compile("gpu_alloc");

    cc::Build::new()
        .cuda(true)
        .file("src/matmul.cu")
        .compile("matmul");

    cc::Build::new()
        .cuda(true)
        .file("src/rmsnorm.cu")
        .compile("rmsnorm");

    cc::Build::new()
        .cuda(true)
        .file("src/rope.cu")
        .compile("rope");

    let cutlass_include = fetch_cutlass();

    // FlashAttention-2's kernels -- see vendor/flash_attn/README.md for the
    // full picture of what's vendored and why. Only the head_dim=128,
    // bf16, causal variant is compiled (matching Qwen2.5-Coder-7B-Instruct's
    // config), both the plain and split-KV kernel instantiations (the
    // latter is what decode -- one new query token against a long cached
    // KV sequence -- actually dispatches to).
    cc::Build::new()
        .cuda(true)
        .include("vendor/flash_attn")
        .include("vendor/stubs")
        .include(&cutlass_include)
        .define("FLASHATTENTION_DISABLE_DROPOUT", None)
        .flag("-std=c++17") // CUTLASS 3.x / CuTe requires C++17
        .flag("--expt-relaxed-constexpr") // CUTLASS's heavy constexpr use needs this nvcc flag
        .flag("-gencode")
        .flag("arch=compute_86,code=sm_86") // A40 = compute capability 8.6 -- VERIFY against the pod's actual GPU (`nvidia-smi --query-gpu=compute_cap --format=csv`) before trusting this
        .file("vendor/flash_attn/flash_fwd_hdim128_bf16_causal_sm80.cu")
        .file("vendor/flash_attn/flash_fwd_split_hdim128_bf16_causal_sm80.cu")
        .file("src/attention.cu")
        .compile("attention");

    // Re-run this build script if the CUDA source changes, not on every
    // build regardless of changes.
    println!("cargo:rerun-if-changed=src/gpu_alloc.cu");
    println!("cargo:rerun-if-changed=src/matmul.cu");
    println!("cargo:rerun-if-changed=src/rmsnorm.cu");
    println!("cargo:rerun-if-changed=src/rope.cu");
    println!("cargo:rerun-if-changed=src/attention.cu");
    println!("cargo:rerun-if-changed=vendor/flash_attn");

    // cuBLAS is a separate shared library from the CUDA runtime itself --
    // linking it explicitly is required for cublasCreate/cublasSgemm/etc.
    // to resolve at link time, the same way gpu_alloc_buffer resolves
    // against cudaMalloc via the CUDA runtime cc::Build already links.
    println!("cargo:rustc-link-lib=cublas");
}

/// Download and extract CUTLASS's header-only `include/` tree into OUT_DIR
/// (cargo's designated scratch directory for build-script output -- already
/// gitignored as part of `target/`, never committed) at the pinned commit
/// above. NOT vendored into the repo directly like flash_attn/ is: see
/// vendor/flash_attn/README.md for why (27MB/805 files is too much foreign
/// source to commit for a single header-only dependency). Skips the
/// download if a previous build already fetched it.
fn fetch_cutlass() -> PathBuf {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cutlass_dir = out_dir.join(format!("cutlass-{CUTLASS_COMMIT}"));
    let include_dir = cutlass_dir.join("include");

    if !include_dir.exists() {
        let tarball = out_dir.join("cutlass.tar.gz");
        let status = Command::new("curl")
            .args(["-sL", "-o"])
            .arg(&tarball)
            .arg(format!(
                "https://codeload.github.com/NVIDIA/cutlass/tar.gz/{CUTLASS_COMMIT}"
            ))
            .status()
            .expect("failed to run curl -- is it installed and is there network access?");
        assert!(status.success(), "curl failed to download CUTLASS");

        let status = Command::new("tar")
            .arg("-xzf")
            .arg(&tarball)
            .arg("-C")
            .arg(&out_dir)
            .status()
            .expect("failed to run tar");
        assert!(status.success(), "tar failed to extract CUTLASS");
    }

    include_dir
}
