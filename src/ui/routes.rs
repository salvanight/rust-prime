//! # Web Server Routes for File Uploader UI
//!
//! This module defines the Actix web server routes and handlers for the file
//! uploading and inspection user interface. It includes routes to serve the main
//! HTML page and to handle file uploads.

use actix_web::{web, App, HttpServer, Responder, HttpResponse, Error};
use actix_files::NamedFile;
use std::path::PathBuf; // Though PathBuf is imported, it's not explicitly used in the provided snippet. Will keep for now.
use actix_multipart::Multipart;
use futures_util::TryStreamExt;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;

/// Serves the main HTML page (`index.html`) for the file uploader UI.
///
/// This function handles GET requests to the root path (`/`).
/// It asynchronously opens and returns the `index.html` file.
pub async fn index() -> impl Responder { // Made public
    NamedFile::open_async("./src/ui/index.html")
        .await
        .expect("index.html should be present in src/ui/") // .unwrap() can panic, added expect for clarity.
}

/// Handles multipart file uploads from the UI.
///
/// This function processes POST requests to the `/upload` path. It expects
/// `multipart/form-data` containing one or more files.
///
/// For each uploaded file, it attempts to:
/// 1. Identify if it's a `.safetensors` model weights file or a `tokenizer.json` file.
/// 2. Parse the file to extract relevant metadata:
///    - For `.safetensors`: Tensor names and their shapes.
///    - For `tokenizer.json`: A preview of the JSON content.
/// 3. Collect any processing errors or skip unsupported files.
///
/// # Memory Considerations:
/// - Each file's content is currently read entirely into a `Vec<u8>` in memory.
///   For very large files (especially `.safetensors`), this can lead to high peak RAM usage.
///   More advanced solutions like memory-mapping temporary files could mitigate this but add complexity.
/// - The `safetensors` crate itself is efficient for metadata extraction once the data is in memory,
///   as it uses views into the byte slice.
/// - `tokenizer.json` files are typically small, making the full read and `serde_json::Value` DOM
///   parsing generally acceptable.
///
/// # Returns
/// An `HttpResponse` which is one of:
/// - **200 OK**: If files were processed (even if some were skipped). The body contains HTML
///   displaying information about successfully processed files and/or messages about skipped files.
///   If no compatible files are processed, a relevant message is shown.
/// - **400 Bad Request**: If any file processing results in a critical error (e.g., corrupted
///   SafeTensors file, invalid JSON). The body contains HTML detailing these errors,
///   alongside information about any files that might have been processed successfully before the error.
/// - Other Actix web error types might be returned by the framework itself on lower-level issues.
pub async fn upload_files(mut payload: Multipart) -> Result<HttpResponse, Error> { // Made public
    let mut uploaded_files_info = HashMap::new();
    let mut processing_errors = Vec::new();
    // Flag to track if any user-facing error (parsing, wrong type) occurred.
    // This helps decide if the overall response should be OK or Bad Request.
    let mut any_critical_error_occurred = false;

    while let Some(item_result) = payload.try_next().await {
        let mut field = match item_result {
            Ok(field) => field,
            Err(e) => {
                processing_errors.push(format!("Error reading multipart stream: {}", e));
                any_critical_error_occurred = true; // Error in stream reading is critical
                continue;
            }
        };

        let content_disposition = field.content_disposition();
        let filename = content_disposition.get_filename().unwrap_or("unknown_file").to_string();

        // Memory consideration: The entire file for this field is read into `file_bytes`.
        // (See previous comments on this topic)
        let mut file_bytes = Vec::new();
        let mut field_chunk_error = false;
        while let Some(chunk_result) = field.try_next().await {
            match chunk_result {
                Ok(chunk) => file_bytes.extend_from_slice(&chunk),
                Err(e) => {
                    let err_msg = format!("Error reading data for file '{}': {}", filename, e);
                    processing_errors.push(err_msg);
                    any_critical_error_occurred = true; // Error reading chunks is critical
                    field_chunk_error = true;
                    break;
                }
            }
        }

        if field_chunk_error {
            continue;
        }

        // Process the collected file bytes using the helper function
        match process_single_file(&filename, file_bytes).await {
            Ok((processed_filename, info_map)) => {
                // Check if it was a skip, which is reported as an Err by process_single_file if it adds to processing_errors.
                // This logic is now simplified: Ok means successfully processed.
                uploaded_files_info.insert(processed_filename, info_map);
            }
            Err(err_msg) => {
                // Errors from process_single_file can be parsing errors or "skipped file" messages.
                // We treat any such message as something to report to the user.
                // If the error indicates a true parsing failure vs. a skip, that could influence any_critical_error_occurred.
                // The current process_single_file returns Err for skips too.
                if err_msg.contains("Invalid") || err_msg.contains("corrupted") { // Heuristic for critical errors
                    any_critical_error_occurred = true;
                }
                processing_errors.push(err_msg);
            }
        }
    }

    // Determine overall response status.
    // If there were critical errors (parsing, stream read), it should be a Bad Request.
    // If only skips occurred, this might also be a Bad Request based on current logic,
    // or could be an OK with a list of issues.
    // The `any_critical_error_occurred` helps distinguish.
    // However, the current HTML generation logic for errors simply lists all `processing_errors`.
    // If `processing_errors` is not empty, it returns BadRequest.
    // This behavior is kept for now.
    // it should be a Bad Request. If files were just skipped (unsupported type),
    // it's more of a partial success, so OK status might still be appropriate,
    // but the message should clearly indicate skipped files.
    // The current logic pushes skipped messages to `processing_errors` and then returns BadRequest
    // if `processing_errors` is not empty. This means skipped files also lead to BadRequest.
    // This can be adjusted if skipped files should not cause a "request failure" status.
    // For now, any entry in `processing_errors` (including skips) makes it a "Bad Request" effectively.

    if !processing_errors.is_empty() {
        let mut error_html_response = String::new();
        error_html_response.push_str("<div class=\"error-message\">");
        error_html_response.push_str("<h2>File Processing Issues:</h2><ul>");
        for err in &processing_errors { // Borrow processing_errors
            error_html_response.push_str(&format!("<li>{}</li>", err));
        }
        error_html_response.push_str("</ul></div>");

        if !uploaded_files_info.is_empty() {
             error_html_response.push_str("<hr><h3>Successfully Processed Files (if any):</h3>");
        }
        // Also include info about successfully processed files if any
        // This part is similar to the success case but appended to the error message
         for (filename, info) in &uploaded_files_info {
            error_html_response.push_str(&format!("<h4>{}</h4>", filename));
            if filename.ends_with(".safetensors") {
                error_html_response.push_str("<ul>");
                for (name, shape) in info {
                    error_html_response.push_str(&format!("<li>Tensor: {}, Shape: {}</li>", name, shape));
                }
                error_html_response.push_str("</ul>");
            } else if filename.ends_with("tokenizer.json") {
                error_html_response.push_str("<pre>");
                for (key, value) in info {
                    if key == "content_preview" {
                        error_html_response.push_str(&value.escape_default().to_string());
                    }
                }
                error_html_response.push_str("</pre>");
            }
        }
        return Ok(HttpResponse::BadRequest().content_type("text/html").body(error_html_response));
    }

    let mut html_response = String::new();
    if uploaded_files_info.is_empty() {
        html_response.push_str("<p>No compatible files were uploaded or processed successfully.</p>");
    } else {
        html_response.push_str("<h2>Uploaded File Information:</h2>");
        for (filename, info) in uploaded_files_info {
            html_response.push_str(&format!("<h3>{}</h3>", filename));
            if filename.ends_with(".safetensors") {
                html_response.push_str("<ul>");
                for (name, shape) in info {
                    html_response.push_str(&format!("<li>Tensor: {}, Shape: {}</li>", name, shape));
                }
                html_response.push_str("</ul>");
            } else if filename.ends_with("tokenizer.json") {
                html_response.push_str("<pre>");
                for (key, value) in info {
                    if key == "content_preview" {
                         html_response.push_str(&value.escape_default().to_string());
                    }
                }
                html_response.push_str("</pre>");
            }
        }
    }
    Ok(HttpResponse::Ok().content_type("text/html").body(html_response))
}

/// Initializes and runs the Actix web server.
///
/// The server is configured with routes for the main page (`/`) and file uploads (`/upload`).
/// It binds to `127.0.0.1:8080` by default.
///
/// # Returns
/// A `std::io::Result<()>` which is `Ok(())` if the server runs successfully,
/// or an `Err` if there's an issue binding to the port or starting the server.
pub async fn run_server() -> std::io::Result<()> {
    println!("Starting server at http://127.0.0.1:8080/"); // Added a startup message
    HttpServer::new(|| {
        App::new()
            // Serve the main index.html page at the root
            .route("/", web::get().to(index))
            // Handle file uploads at /upload
            .route("/upload", web::post().to(upload_files))
    })
    .bind(("127.0.0.1", 8080))?
}

// Helper function to process a single uploaded file
// Returns Ok((filename, info_map)) on successful parsing, where info_map includes a "_type" field.
// Returns Err(error_message_string) for parsing errors or if the file type is skipped.
async fn process_single_file(
    filename: &str, // Borrow filename
    file_bytes: Vec<u8>,
) -> Result<(String, HashMap<String, String>), String> {
    if filename.ends_with(".safetensors") {
        // (Memory consideration comments remain applicable here)
        println!("Processing SafeTensors file: {}", filename);
        match SafeTensors::deserialize(&file_bytes) {
            Ok(tensors) => {
                let mut tensor_info = HashMap::new();
                for (name, view) in tensors.tensors() {
                    tensor_info.insert(name.clone(), format!("{:?}", view.shape()));
                }
                tensor_info.insert("_type".to_string(), "safetensor".to_string()); // For HTML generation
                println!("Successfully processed SafeTensors file: {}", filename);
                Ok((filename.to_string(), tensor_info))
            }
            Err(e) => {
                let error_msg = format!(
                    "<strong>File: '{}'</strong> - Invalid SafeTensors format. Details: {}. Please ensure the file is not corrupted and conforms to the SafeTensors specification.",
                    filename, e
                );
                eprintln!("Error deserializing SafeTensors file {}: {}", filename, e);
                Err(error_msg)
            }
        }
    } else if filename.ends_with("tokenizer.json") {
        // (Memory consideration comments remain applicable here)
        println!("Processing tokenizer.json file: {}", filename);
        match serde_json::from_slice::<Value>(&file_bytes) {
            Ok(json_value) => {
                let mut tokenizer_details = HashMap::new();
                tokenizer_details.insert("content_preview".to_string(), format!("{:.200}", json_value.to_string()));
                tokenizer_details.insert("_type".to_string(), "tokenizer".to_string()); // For HTML generation
                println!("Successfully processed tokenizer.json file: {}", filename);
                Ok((filename.to_string(), tokenizer_details))
            }
            Err(e) => {
                let error_msg = format!(
                    "<strong>File: '{}'</strong> - Invalid JSON format in tokenizer file. Details: {}. Please check for syntax errors like missing commas or brackets.",
                    filename, e
                );
                eprintln!("Error deserializing tokenizer.json file {}: {}", filename, e);
                Err(error_msg)
            }
        }
    } else {
        let skipped_msg = format!(
            "<strong>File: '{}'</strong> - Skipped. This file type is not supported. Please upload only .safetensors or tokenizer.json files.",
            filename
        );
        // This is treated as an "error" for reporting purposes to ensure the user is notified.
        Err(skipped_msg)
    }
}

/// Initializes and runs the Actix web server.
    .run()
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, web::Bytes, http::StatusCode};
    use actix_multipart::{Field, Multipart};
    use futures_util::stream::iter;
    use std::io::Write;
    use safetensors::serialize;
    use std::collections::HashMap;
    use actix_http::header::{CONTENT_TYPE, HeaderValue, HeaderName, DispositionType, ContentDisposition};


    // Helper function to create a multipart payload with a single file
    fn create_multipart_payload(filename: &str, content: Vec<u8>) -> Multipart {
        let field = Field::new(
            HeaderName::from_static("content-disposition"),
            HeaderValue::from_str(&format!("form-data; name=\"files\"; filename=\"{}\"", filename)).unwrap(),
            iter(vec![Ok(Bytes::from(content))])
        );
        Multipart::new(
            &HeaderValue::from_static("multipart/form-data; boundary=--boundary"),
            iter(vec![Ok(field)])
        )
    }

    // Helper function to create a multipart payload with multiple files
    fn create_multipart_payload_multiple(files: Vec<(&str, Vec<u8>)>) -> Multipart {
        let mut fields = Vec::new();
        for (filename, content) in files {
            let field = Field::new(
                HeaderName::from_static("content-disposition"),
                HeaderValue::from_str(&format!("form-data; name=\"files\"; filename=\"{}\"", filename)).unwrap(),
                iter(vec![Ok(Bytes::from(content))])
            );
            fields.push(Ok(field));
        }
        Multipart::new(
            &HeaderValue::from_static("multipart/form-data; boundary=--boundary"),
            iter(fields)
        )
    }


    #[actix_rt::test]
    async fn test_upload_tokenizer_json_success() {
        let tokenizer_content = r#"{"name": "test_tokenizer"}"#.as_bytes().to_vec();
        let payload = create_multipart_payload("tokenizer.json", tokenizer_content);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<h3>tokenizer.json</h3>"));
        assert!(body_str.contains(r#"name&quot;: &quot;test_tokenizer&quot;"#));
    }

    #[actix_rt::test]
    async fn test_upload_safetensors_success() {
        // Create a minimal valid safetensors file (empty metadata, no tensors)
        // More complex tensor data could be added here if needed.
        let empty_tensors: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
        let safetensor_bytes = serialize(&empty_tensors, &None).expect("Failed to serialize empty safetensors");

        let payload = create_multipart_payload("dummy.safetensors", safetensor_bytes);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<h3>dummy.safetensors</h3>"));
        // Check for "<ul></ul>" as there are no tensors, or specific tensor info if you add some
        assert!(body_str.contains("<ul></ul>"));
    }

    #[actix_rt::test]
    async fn test_upload_tokenizer_json_corrupted() {
        let corrupted_content = r#"{"name": "test_tokenizer",}"#.as_bytes().to_vec(); // Invalid JSON
        let payload = create_multipart_payload("tokenizer_corrupted.json", corrupted_content);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'tokenizer_corrupted.json'</strong> - Invalid JSON format in tokenizer file."));
    }

    #[actix_rt::test]
    async fn test_upload_safetensors_corrupted() {
        let corrupted_bytes = b"this is not a valid safetensors file".to_vec();
        let payload = create_multipart_payload("corrupted.safetensors", corrupted_bytes);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'corrupted.safetensors'</strong> - Invalid SafeTensors format."));
    }

    #[actix_rt::test]
    async fn test_upload_unknown_file_type() {
        let other_content = "This is a plain text file.".as_bytes().to_vec();
        let payload = create_multipart_payload("other.txt", other_content);

        let resp = upload_files(payload).await.unwrap();
        // Now expects BAD_REQUEST because skipped files are treated as processing issues.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'other.txt'</strong> - Skipped. This file type is not supported."));
        // Check that there's no "Successfully Processed Files" section if only a skip occurs.
        assert!(!body_str.contains("<hr><h3>Successfully Processed Files (if any):</h3>"));
    }

    #[actix_rt::test]
    async fn test_upload_mixed_files_valid_and_invalid() {
        let valid_tokenizer_content = r#"{"name": "valid_tokenizer"}"#.as_bytes().to_vec();
        let corrupted_safetensor_bytes = b"invalid safetensor data".to_vec();
        let other_content = "plain text".as_bytes().to_vec();

        let files = vec![
            ("tokenizer.json", valid_tokenizer_content),
            ("bad.safetensors", corrupted_safetensor_bytes),
            ("notes.txt", other_content),
        ];
        let payload = create_multipart_payload_multiple(files);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST); // Because one file is bad

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<div class=\"error-message\">"));
        assert!(body_str.contains("<h2>File Processing Issues:</h2>"));
        assert!(body_str.contains("<strong>File: 'bad.safetensors'</strong> - Invalid SafeTensors format."));
        assert!(body_str.contains("<strong>File: 'notes.txt'</strong> - Skipped. This file type is not supported."));

        assert!(body_str.contains("<hr><h3>Successfully Processed Files (if any):</h3>"));
        assert!(body_str.contains("<h3>tokenizer.json</h3>"));
        assert!(body_str.contains(r#"name&quot;: &quot;valid_tokenizer&quot;"#));

        // Ensure notes.txt was skipped and not listed as processed or error
        assert!(!body_str.contains("<h3>notes.txt</h3>"));
    }
     #[actix_rt::test]
    async fn test_upload_multiple_valid_files() {
        let tokenizer_content = r#"{"name": "multi_tokenizer"}"#.as_bytes().to_vec();
        let empty_tensors: HashMap<String, safetensors::tensor::TensorView> = HashMap::new();
        let safetensor_bytes = serialize(&empty_tensors, &None).expect("Failed to serialize empty safetensors");

        let files = vec![
            ("good_tokenizer.json", tokenizer_content),
            ("good.safetensors", safetensor_bytes),
        ];
        let payload = create_multipart_payload_multiple(files);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        assert!(body_str.contains("<h2>Uploaded File Information:</h2>"));
        assert!(body_str.contains("<h3>good_tokenizer.json</h3>"));
        assert!(body_str.contains(r#"name&quot;: &quot;multi_tokenizer&quot;"#));
        assert!(body_str.contains("<h3>good.safetensors</h3>"));
        assert!(body_str.contains("<ul></ul>")); // For empty safetensor
    }
}
