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

fn main() {
    cc::Build::new()
        .cuda(true)
        .file("src/gpu_alloc.cu")
        .compile("gpu_alloc");

    // Re-run this build script if the CUDA source changes, not on every
    // build regardless of changes.
    println!("cargo:rerun-if-changed=src/gpu_alloc.cu");
}
