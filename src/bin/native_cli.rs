// src/main.rs
use rust_transformers_gpt2::native::runtime_interface;
use rust_transformers_gpt2::ui::routes::run_server;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Remove the placeholder call
    // rust_native_transformer::runtime_interface::run_cli_placeholder();

    // Commenting out the existing CLI logic for now
    // if let Err(e) = runtime_interface::run_cli() {
    //     eprintln!("Application error: {}", e);
    //     // Optionally, print the full chain of errors if e.source() is available
    //     let mut current_err: Option<&(dyn std::error::Error + 'static)> = e.source();
    //     while let Some(source) = current_err {
    //         eprintln!("Caused by: {}", source);
    //         current_err = source.source();
    //     }
    //     std::process::exit(1);
    // }

    println!("Starting server at http://127.0.0.1:8080/");
    run_server().await
}
