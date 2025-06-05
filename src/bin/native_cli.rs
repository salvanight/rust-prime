// src/main.rs
use rust_transformers_gpt2::native::runtime_interface;
use rust_transformers_gpt2::ui::routes::run_server;
use clap::Parser;

/// Simple CLI to run the experimental UI server.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Port number for the web server
    #[clap(short, long, value_parser, default_value_t = 8080)]
    port: u16,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Args::parse();

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

    // The println! message will be updated in run_server in routes.rs
    // println!("Starting server at http://127.0.0.1:{}", args.port);
    run_server(args.port).await
}
