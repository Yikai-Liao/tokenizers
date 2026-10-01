fn main() {
    use std::process::Command;
    let out = std::env::var("OUT_DIR").unwrap();
    assert!(Command::new("cc").args(["-O3", "-DNDEBUG", "-c", "vendor/radixsort_permuted.c", "-o", &format!("{out}/radsort.o")]).status().unwrap().success());
    assert!(Command::new("cc").args(["-O3", "-c", "vendor/wrapper.c", "-o", &format!("{out}/wrapper.o")]).status().unwrap().success());
    assert!(Command::new("ar").args(["rcs", &format!("{out}/libradsort.a"), &format!("{out}/radsort.o"), &format!("{out}/wrapper.o")]).status().unwrap().success());
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-lib=static=radsort");
    println!("cargo:rerun-if-changed=vendor/radixsort_permuted.c");
    println!("cargo:rerun-if-changed=vendor/wrapper.c");
}
