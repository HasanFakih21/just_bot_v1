use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

const BASE_URL: &str = "https://github.com/HasanFakih21/JustBot-Networks/releases/download/Networks";
const NETWORK_NAME: &str = "595c4ib-1024.nnue";

fn main() {
    set_model_env_var();
    let network_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(NETWORK_NAME);
    if !network_path.exists() && env::var("EVALFILE").is_err() {
        download_netowrk();
    }

    println!("cargo::rerun-if-changed={}", network_path.display());
    println!("cargo::rerun-if-env-changed=EVALFILE")
}

fn set_model_env_var() {
    let net = env::var("EVALFILE").unwrap_or(NETWORK_NAME.to_string());
    let mut path = PathBuf::from(net);

    if path.is_relative() {
        path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    }

    println!("cargo::rustc-env=MODEL={}", path.display());
}

fn download_netowrk() {
    let output = Command::new("curl")
        .args(["-s", "-O", "-L"])
        .arg(format!("{BASE_URL}/{NETWORK_NAME}"))
        .output()
        .expect("Error executing 'curl'!");

    if !output.status.success() {
        panic!("Error downloading network!");
    }
}
