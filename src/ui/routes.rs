use actix_web::{web, App, HttpServer, Responder, HttpResponse, Error};
use actix_files::NamedFile;
use std::path::PathBuf;
use actix_multipart::Multipart;
use futures_util::TryStreamExt;
use safetensors::SafeTensors;
use serde_json::Value;
use std::collections::HashMap;

pub async fn index() -> impl Responder { // Made public
    NamedFile::open_async("./src/ui/index.html").await.unwrap()
}

pub async fn upload_files(mut payload: Multipart) -> Result<HttpResponse, Error> { // Made public
    let mut uploaded_files_info = HashMap::new();
    let mut processing_errors = Vec::new();

    while let Some(item_result) = payload.try_next().await {
        let mut field = match item_result {
            Ok(field) => field,
            Err(e) => {
                processing_errors.push(format!("Error reading multipart item: {}", e));
                continue;
            }
        };
        let content_disposition = field.content_disposition();
        let filename = content_disposition.get_filename().unwrap_or("unknown_file").to_string();

        let mut file_bytes = Vec::new();
        while let Some(chunk_result) = field.try_next().await {
            match chunk_result {
                Ok(chunk) => file_bytes.extend_from_slice(&chunk),
                Err(e) => {
                    processing_errors.push(format!("Error reading chunk for file {}: {}", filename, e));
                    // Skip to next file if chunk reading fails
                    continue;
                }
            }
        }

        if filename.ends_with(".safetensors") {
            println!("Processing SafeTensors file: {}", filename);
            match SafeTensors::deserialize(&file_bytes) {
                Ok(tensors) => {
                    let mut tensor_info = HashMap::new();
                    for (name, view) in tensors.tensors() {
                        tensor_info.insert(name.clone(), format!("{:?}", view.shape()));
                    }
                    uploaded_files_info.insert(filename.clone(), tensor_info);
                    println!("Successfully processed SafeTensors file: {}", filename);
                }
                Err(e) => {
                    let error_msg = format!("Error deserializing SafeTensors file '{}': {}", filename, e);
                    eprintln!("{}", error_msg);
                    processing_errors.push(error_msg);
                }
            }
        } else if filename.ends_with("tokenizer.json") {
            println!("Processing tokenizer.json file: {}", filename);
            match serde_json::from_slice::<Value>(&file_bytes) {
                Ok(json_value) => {
                    let mut tokenizer_details = HashMap::new();
                    tokenizer_details.insert("content_preview".to_string(), format!("{:.200}", json_value.to_string())); // Increased preview length
                    uploaded_files_info.insert(filename.clone(), tokenizer_details);
                    println!("Successfully processed tokenizer.json file: {}", filename);
                }
                Err(e) => {
                    let error_msg = format!("Error deserializing tokenizer.json file '{}': {}", filename, e);
                    eprintln!("{}", error_msg);
                    processing_errors.push(error_msg);
                }
            }
        } else {
            println!("Skipping non-SafeTensors/tokenizer.json file: {}", filename);
        }
    }

    if !processing_errors.is_empty() {
        let mut error_html_response = String::new();
        error_html_response.push_str("<h2>File Processing Errors:</h2><ul>");
        for err in processing_errors {
            error_html_response.push_str(&format!("<li>{}</li>", err));
        }
        error_html_response.push_str("</ul>");
        if !uploaded_files_info.is_empty(){
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


pub async fn run_server() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(index))
            .route("/upload", web::post().to(upload_files))
    })
    .bind(("127.0.0.1", 8080))?
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
        assert!(body_str.contains("File Processing Errors"));
        assert!(body_str.contains("Error deserializing tokenizer.json file 'tokenizer_corrupted.json'"));
    }

    #[actix_rt::test]
    async fn test_upload_safetensors_corrupted() {
        let corrupted_bytes = b"this is not a valid safetensors file".to_vec();
        let payload = create_multipart_payload("corrupted.safetensors", corrupted_bytes);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("File Processing Errors"));
        assert!(body_str.contains("Error deserializing SafeTensors file 'corrupted.safetensors'"));
    }

    #[actix_rt::test]
    async fn test_upload_unknown_file_type() {
        let other_content = "This is a plain text file.".as_bytes().to_vec();
        let payload = create_multipart_payload("other.txt", other_content);

        let resp = upload_files(payload).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK); // Should be OK, but with "no compatible files" message

        let body = test::body_to_bytes(resp.into_body()).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("<p>No compatible files were uploaded or processed successfully.</p>"));
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

        assert!(body_str.contains("File Processing Errors"));
        assert!(body_str.contains("Error deserializing SafeTensors file 'bad.safetensors'"));

        assert!(body_str.contains("Successfully Processed Files (if any)"));
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
