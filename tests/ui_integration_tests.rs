use actix_web::{web, App, HttpServer, Responder};
use rust_transformers_gpt2::ui::routes::run_server; // Assuming run_server is pub
use rust_transformers_gpt2::ui::mod as ui_module; // To get access to routes for App configuration
use std::fs;
use std::path::Path;
use actix_rt;
use awc::Client;
use awc::http::header;
use actix_multipart::Multipart; // This is for server-side, for client use awc::multipart
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

// Helper to read file bytes
fn read_test_file(file_path: &str) -> Vec<u8> {
    fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join(file_path)).expect(&format!("Failed to read test file: {}", file_path))
}

#[actix_rt::test]
async fn test_upload_endpoint_integration() {
    // Pick a free port or use a fixed one if sure it's available
    let server_addr = "127.0.0.1:0"; // Use port 0 to let OS pick a free port
    let (tx, rx) = mpsc::channel();

    let server_handle = thread::spawn(move || {
        let server_future = HttpServer::new(|| {
            App::new().service(
                web::scope("") // Ensure routes are registered as they are in main
                    .route("/", web::get().to(ui_module::routes::index)) // Assuming index is pub or re-export
                    .route("/upload", web::post().to(ui_module::routes::upload_files)) // Assuming upload_files is pub
            )
        })
        .bind(server_addr)
        .unwrap()
        .run();

        let server_handle = server_future.handle();
        tx.send(server_handle).unwrap(); // Send back the server handle

        let rt = actix_rt::System::new();
        rt.block_on(server_future).unwrap();
    });

    // Wait for the server to start by waiting for the handle
    let server_handle_received = rx.recv_timeout(Duration::from_secs(10)).expect("Server did not start in time");

    // Get the actual address the server bound to
    // This is a bit tricky as HttpServer::bind returns a Vec of Sockets
    // For simplicity, we'll assume the first one is what we want if port 0 was used.
    // However, Actix test server is usually better for getting the bound address.
    // For now, let's assume we know the port or use a fixed one for tests if this part is too complex.
    // Let's hardcode a test port for now and ensure `run_server` uses it or can be configured.
    // Re-evaluating: `run_server` from `src/ui/routes.rs` binds to "127.0.0.1:8080".
    // We should use that, or make it configurable. For this test, let's assume 8080.
    // The separate thread for the server is better for true integration tests.

    let test_server_addr = "127.0.0.1:8088"; // Using a different port for this test server
    let (tx_test_addr, rx_test_addr) = mpsc::channel();


    let test_server_thread = thread::spawn(move || {
        let sys = actix_rt::System::new();
        let server = HttpServer::new(|| {
            App::new()
                .route("/upload", web::post().to(ui_module::routes::upload_files))
        })
        .bind(test_server_addr)
        .unwrap();
        let server_handle = server.handle();
        tx_test_addr.send((server.handle(), server_addr.to_string())).unwrap(); // Send handle and actual bound address

        sys.block_on(server.run()).unwrap();
    });

    let (test_srv_handle, _bound_addr_str) = rx_test_addr.recv_timeout(Duration::from_secs(5)).expect("Test server did not start");


    let client = Client::default();

    let valid_tokenizer_bytes = read_test_file("src/ui/test_data/tokenizer.json");
    let corrupted_safetensor_bytes = read_test_file("src/ui/test_data/dummy_corrupted.safetensors"); // Using the placeholder
    let other_file_bytes = read_test_file("src/ui/test_data/other_file.txt");

    let form = awc::multipart::Form::new()
        .add_field("files", "tokenizer.json", "application/json", valid_tokenizer_bytes)
        .add_field("files", "corrupted.safetensors", "application/octet-stream", corrupted_safetensor_bytes)
        .add_field("files", "other.txt", "text/plain", other_file_bytes);

    let request_url = format!("http://{}/upload", test_server_addr);
    let response = client.post(&request_url)
        .content_type(form.content_type()) // Important: Set the correct multipart content type
        .send_body(form)
        .await
        .expect("Failed to send request");

    assert_eq!(response.status().as_u16(), 400, "Expected Bad Request due to corrupted and skipped files");

    let body_bytes = response.body().await.expect("Failed to read response body");
    let body_str = std::str::from_utf8(&body_bytes).expect("Response body was not valid UTF-8");

    // Check for the main error wrapper and heading
    assert!(body_str.contains("<div class=\"error-message\">"), "Body should contain error-message div");
    assert!(body_str.contains("<h2>File Processing Issues:</h2>"), "Body should contain processing issues header");

    // Check for specific error messages
    assert!(
        body_str.contains("<strong>File: 'corrupted.safetensors'</strong> - Invalid SafeTensors format."),
        "Body should mention corrupted safetensor with new formatting"
    );
    assert!(
        body_str.contains("<strong>File: 'other.txt'</strong> - Skipped. This file type is not supported."),
        "Body should mention skipped other.txt file with new formatting"
    );

    // Check for the section detailing successfully processed files (if any)
    assert!(
        body_str.contains("<hr><h3>Successfully Processed Files (if any):</h3>"),
        "Body should have success section for valid files"
    );
    assert!(
        body_str.contains("<h4>tokenizer.json</h4>"), // Changed from h3 to h4 in error report for successful items
        "Body should contain info for valid tokenizer"
    );
    // Ensure the content of tokenizer.json is still there (simplified check)
    assert!(
        body_str.contains("content_preview"),
        "Body should contain tokenizer preview content"
    );


    // Stop the server
    test_srv_handle.stop(true).await;
    test_server_thread.join().expect("Test server thread panicked");
}
