use std::env;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor_src = manifest_dir.join("../../vendor/sqlite-vector/src");
    let vendor_libs = manifest_dir.join("../../vendor/sqlite-vector/libs");

    // sqlite3.h is vendored alongside sqlite-vector sources.
    // This avoids depending on DEP_SQLITE3_INCLUDE from libsqlite3-sys,
    // which is not reliably available when build scripts run in parallel.
    let sqlite_include = &vendor_src;

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    assert!(
        target_os == "macos",
        "Milestone 1 only supports macOS. Target OS: {target_os}"
    );

    // Common files: sqlite-vector.c + distance-cpu.c
    let mut common = cc::Build::new();
    common
        .file(vendor_src.join("sqlite-vector.c"))
        .file(vendor_src.join("distance-cpu.c"))
        .include(&vendor_src)
        .include(&vendor_libs)
        .include(sqlite_include)
        .define("SQLITE_CORE", None)
        .warnings(false)
        .opt_level(2);

    match target_arch.as_str() {
        "aarch64" => {
            // Apple Silicon: NEON is baseline, no extra flags needed.
            common.file(vendor_src.join("distance-neon.c"));
            common.compile("sqlite_vector");
        }
        "x86_64" => {
            // Intel Mac: SSE2 is baseline on all x86_64 CPUs.
            // Provide stubs that fall back to SSE2 for the missing
            // AVX2/AVX512 init functions. At runtime, if the CPU supports
            // AVX2 (or AVX512), distance-cpu.c calls init_distance_functions_avx2
            // (or avx512) which would skip SSE2 init entirely. Our stubs
            // redirect to SSE2 so the search still works correctly.
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let stub_path = out_dir.join("distance_stubs.c");
            let mut stub = std::fs::File::create(&stub_path).unwrap();
            writeln!(stub, "extern void init_distance_functions_sse2(void);").unwrap();
            writeln!(
                stub,
                "void init_distance_functions_avx2(void) {{ init_distance_functions_sse2(); }}"
            )
            .unwrap();
            writeln!(
                stub,
                "void init_distance_functions_avx512(void) {{ init_distance_functions_sse2(); }}"
            )
            .unwrap();

            common.file(vendor_src.join("distance-sse2.c"));
            common.file(&stub_path);
            common.compile("sqlite_vector");
        }
        _ => {
            panic!("Unsupported architecture: {target_arch}");
        }
    }

    println!("cargo::rerun-if-changed=../../vendor/sqlite-vector/src");
    println!("cargo::rerun-if-changed=../../vendor/sqlite-vector/libs/fp16");
}
