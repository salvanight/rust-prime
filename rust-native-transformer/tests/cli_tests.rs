use std::process::Command;
use std::str;

// Helper to find the CLI executable
fn get_cli_path() -> String {
    // Assumes CLI is built in debug mode by `cargo test`
    // Adjust if using release builds for testing, though debug is typical.
    let exe_name = "rust_native_transformer_cli";
    // Path relative to where `cargo test` runs from for integration tests (package root)
    format!("target/debug/{}", exe_name)
}

#[test]
fn test_cli_help_message() {
    let output = Command::new(get_cli_path())
        .arg("--help")
        .output()
        .expect("Failed to execute --help command");

    assert!(output.status.success(), "CLI --help exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).expect("stdout is not valid UTF-8");
    
    // Check for common help text patterns
    assert!(stdout.contains("Usage:"), "Help message should contain 'Usage:'");
    assert!(stdout.contains("Options:"), "Help message should contain 'Options:'");
    assert!(stdout.contains("--model-path"), "Help message should mention --model-path");
}

#[test]
fn test_cli_version_message() {
    let output = Command::new(get_cli_path())
        .arg("--version")
        .output()
        .expect("Failed to execute --version command");

    assert!(output.status.success(), "CLI --version exited with error: {:?}", output);
    let stdout = str::from_utf8(&output.stdout).expect("stdout is not valid UTF-8");
    
    // Assuming version format like "rust_native_transformer_cli 0.1.0"
    // The actual name might be just "rust_native_transformer" if [[bin]] name is used.
    // The executable name from get_cli_path() is rust_native_transformer_cli.
    // Clap usually uses the package name + version from Cargo.toml.
    // Let's check for the package name "rust_native_transformer" and the version "0.1.0".
    assert!(stdout.contains("rust_native_transformer_cli 0.1.0") || stdout.contains("rust_native_transformer 0.1.0"), 
            "Version output did not contain expected package name and version. Output: {}", stdout);
}

#[test]
fn test_cli_missing_required_args() {
    // Required args are: --model-path, --vocab-path, --merges-path, --prompt
    // Test by omitting --model-path
    let output = Command::new(get_cli_path())
        .arg("--vocab-path")
        .arg("dummy_vocab.json")
        .arg("--merges-path")
        .arg("dummy_merges.txt")
        .arg("--prompt")
        .arg("Hello")
        .output()
        .expect("Failed to execute command with missing --model-path");

    assert!(!output.status.success(), "CLI should fail when --model-path is missing. Output: {:?}", output);
    let stderr = str::from_utf8(&output.stderr).expect("stderr is not valid UTF-8");
    
    assert!(stderr.contains("the following required arguments were not provided"), 
            "Stderr should indicate missing arguments. Stderr: {}", stderr);
    assert!(stderr.contains("--model-path <MODEL_PATH>"), 
            "Stderr should specifically mention missing --model-path. Stderr: {}", stderr);
}

#[test]
fn test_cli_invalid_argument_value_max_length() {
    // Test --max-length with a non-integer value
    let output_non_int = Command::new(get_cli_path())
        .args([
            "--model-path", "dummy.safetensors", 
            "--vocab-path", "dummy_vocab.json",
            "--merges-path", "dummy_merges.txt",
            "--prompt", "Hello",
            "--max-length", "not_a_number" 
        ])
        .output()
        .expect("Failed to execute command with invalid --max-length (non-int)");

    assert!(!output_non_int.status.success(), "CLI should fail with non-integer --max-length. Output: {:?}", output_non_int);
    let stderr_non_int = str::from_utf8(&output_non_int.stderr).expect("stderr is not valid UTF-8");
    assert!(stderr_non_int.contains("invalid value 'not_a_number' for '--max-length <MAX_LENGTH>'"), 
            "Stderr should indicate invalid value for --max-length. Stderr: {}", stderr_non_int);
    
    // Test --max-length with a negative value (if clap is set up to parse u32/usize, it will catch this)
    // The `max_length` field in `CliArgs` is `usize`. Clap's default parser for `usize`
    // will reject negative numbers.
    let output_negative = Command::new(get_cli_path())
         .args([
            "--model-path", "dummy.safetensors",
            "--vocab-path", "dummy_vocab.json",
            "--merges-path", "dummy_merges.txt",
            "--prompt", "Hello",
            "--max-length", "-5"
        ])
        .output()
        .expect("Failed to execute command with invalid --max-length (negative)");

    assert!(!output_negative.status.success(), "CLI should fail with negative --max-length. Output: {:?}", output_negative);
    let stderr_negative = str::from_utf8(&output_negative.stderr).expect("stderr is not valid UTF-8");
    // Clap's behavior for a usize argument receiving a value like "-5":
    // It might see "-5" as an attempt to specify an option, not as a value for --max-length.
    assert!(stderr_negative.contains("unexpected argument '-5' found"),
            "Stderr should indicate unexpected argument for --max-length with negative value. Stderr: {}", stderr_negative);
}

#[test]
fn test_cli_no_model_provided_graceful_error() {
    // Provide all required args, but --model-path points to a non-existent file.
    // This tests if the application logic (after parsing) handles the error.
    let output = Command::new(get_cli_path())
        .args([
            "--model-path", "non_existent_model.safetensors",
            // Use actual paths to tokenizer resources to ensure tokenizer loading doesn't fail first
            "--vocab-path", "../resources/tokenizer_data/gpt2/gpt2-vocab.json", 
            "--merges-path", "../resources/tokenizer_data/gpt2/merges.txt",
            "--prompt", "Hello"
            // Default values for config args (--config-n-layer etc.) from CliArgs will be used.
            // The test should fail at model loading.
        ])
        .output()
        .expect("Failed to execute command with non-existent model path");

    assert!(!output.status.success(), "CLI should fail when model file does not exist. Output: {:?}", output);
    let stderr = str::from_utf8(&output.stderr).expect("stderr is not valid UTF-8");

    // The error message comes from `runtime_interface.rs` when `load_safetensors` fails.
    // It's wrapped in `RuntimeError::ModelLoader` and then displayed.
    // The Display impl for ModelLoaderError::IoError(e) prints "IO error: {}", where {} is the io::Error's message.
    // The Debug impl for RuntimeError prints "ModelLoader error: {:?}" for the ModelLoaderError.
    // So, we expect "Application error: ModelLoader error: IoError(Os { code: 2, kind: NotFound, message: "No such file or directory" })"
    // The filename itself isn't part of the formatted error string from Display/Debug of the error enum chain.
    let expected_error_substring = "ModelLoader error: IoError(Os { code: 2, kind: NotFound, message: \"No such file or directory\" })";
    assert!(stderr.contains(expected_error_substring),
            "Stderr should indicate model loading failure. Expected substring: '{}'. Actual Stderr: {}", expected_error_substring, stderr);
}

// Note: The `dummy_vocab.json`, `dummy_merges.txt` files do not need to exist 
// for tests that fail *before* attempting to load them (like missing_required_args, invalid_arg_value).
// For `test_cli_no_model_provided_graceful_error`, they are provided as paths, but the test
// should fail at model loading before it attempts to load these.
// If any test *successfully* parses and proceeds to load vocab/merges, those dummy files would need to exist.
// For these specific tests, we are mostly testing argument parsing and early exit conditions.
