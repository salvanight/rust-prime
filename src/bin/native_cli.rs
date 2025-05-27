// src/main.rs
use rust_transformers_gpt2::native::runtime_interface;

fn main() {
    // Remove the placeholder call
    // rust_native_transformer::runtime_interface::run_cli_placeholder();

    if let Err(e) = runtime_interface::run_cli() {
        eprintln!("Application error: {}", e);
        // Optionally, print the full chain of errors if e.source() is available
        let mut current_err: Option<&(dyn std::error::Error + 'static)> = e.source();
        while let Some(source) = current_err {
            eprintln!("Caused by: {}", source);
            current_err = source.source();
        }
        std::process::exit(1);
    }
}
