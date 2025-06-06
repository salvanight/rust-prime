# Application Architecture

## 1. Overview

This project is a Rust-based application designed for working with transformer models. It currently features two main interaction modes: a native Command Line Interface (CLI) primarily focused on model inference (though this part is less emphasized in the current UI-focused development path), and an experimental web-based User Interface (UI). The web UI allows users to upload and inspect model-related files, specifically `.safetensors` for weights and `tokenizer.json` for tokenizer configurations.

The high-level components are:
- **Command Line Interface (CLI):** The entry point for launching the application, including the web UI server.
- **Web UI Backend:** An Actix web server that serves the frontend and handles file processing requests.
- **Web UI Frontend:** A single-page application built with vanilla JavaScript, HTML, and CSS (Pico.CSS) that runs in the user's browser.
- **Core File Processing & Model Utilities:** A Rust library (primarily in `src/`) that provides functionalities for handling model files, tokenizers, and tensor operations. The UI backend directly uses some of these underlying crates for inspection tasks.

## 2. Component Details

### 2.1. Command Line Interface (CLI)
    - **Responsibilities:**
        - Main entry point for the application (`native_cli` binary).
        - Parses command-line arguments, including the `--port` option for configuring the web UI server's listening port.
        - Initializes and starts the Web UI Backend (Actix web server).
    - **Key Modules/Files:** `src/bin/native_cli.rs`.
    - **Key Dependencies/Technologies:** `clap` (for argument parsing), `actix-web` (to run the server).
    - **Interactions:** Launches the Web UI Backend and passes the configured port number to it. The original CLI functionalities for model inference are currently commented out in `native_cli.rs` to prioritize the UI server.

### 2.2. Web UI Backend (Actix Web Server)
    - **Responsibilities:**
        - Serves the static frontend UI files (HTML, CSS, JS contained within `src/ui/index.html`).
        - Handles API requests from the frontend, primarily for file uploads via a POST request to `/upload`.
        - Processes uploaded files:
            - For `.safetensors` files: Parses metadata to extract tensor names, shapes, data types, and generates data previews.
            - For `tokenizer.json` files: Parses the JSON to provide a content preview.
        - Returns processing results (success or error details) to the frontend as dynamically generated HTML fragments.
    - **Key Modules/Files:** `src/ui/routes.rs` (defines request handlers and file processing logic), `src/ui/mod.rs` (module definition).
    - **Key Dependencies/Technologies:** `actix-web` (web framework), `actix-files` (for serving `index.html`), `actix-multipart` (for handling file uploads), `safetensors` (for parsing `.safetensors` files), `serde_json` (for parsing `tokenizer.json`), `bytemuck` (for casting byte slices for tensor previews), `html-escape` (for safely embedding data into HTML attributes).
    - **Interactions:** Receives HTTP requests from the Web UI Frontend. Responds with the main `index.html` page or HTML content fragments generated from file processing.

### 2.3. Web UI Frontend (Single Page Application)
    - **Responsibilities:**
        - Renders the user interface in the browser.
        - Handles user interactions:
            - File selection via a browse button or drag-and-drop.
            - Triggering file uploads.
            - Clicking on tensor list items to view details.
            - Selecting tensors via checkboxes for comparison.
            - Opening and closing modal dialogs for tensor details and comparison.
        - Sends files to the backend using the `fetch` API.
        - Dynamically updates the DOM with results, error messages, and modal content received from the backend or generated client-side.
    - **Key Modules/Files:** `src/ui/index.html` (contains all HTML structure, CSS styles including Pico.CSS, and all client-side JavaScript logic).
    - **Key Dependencies/Technologies:** HTML5, CSS3 (Pico.CSS for base styling, plus custom styles), Vanilla JavaScript (ES6+ features like async/await, FormData).
    - **Interactions:** Makes HTTP POST requests (for file uploads) to the `/upload` endpoint of the Web UI Backend. Receives HTML fragments from the backend and injects them into the DOM. Manages all client-side views (main list, detail modal, comparison modal) and state (e.g., selected tensors for comparison).

### 2.4. Core File Processing & Model Utilities (Rust Library)
    - **Responsibilities:** This layer is intended to provide the underlying capabilities for loading, interpreting, and working with transformer models and related files. For the current UI inspection features, the UI backend often uses foundational crates directly.
        - **SafeTensors Handling:** Provides the capability to parse `.safetensors` files, extract metadata (tensor names, shapes, dtypes), and access tensor data for previews. The UI backend uses the `safetensors` crate directly for this.
        - **Tokenizer Handling:** Provides capability to parse `tokenizer.json`. The UI backend uses `serde_json` directly for this. The broader project also contains a `TokenizerWrapper` in `src/tokenizer.rs` for more comprehensive tokenizer operations, but this is not directly used by the current UI file inspection features.
        - **Other Model Utilities:** The `src/` directory contains modules for model architecture definitions, tensor operations (`src/tensor_engine.rs`), etc., which are part of the broader project's goal but not directly invoked by the current UI inspection features.
    - **Key Modules/Files:**
        - `src/ui/routes.rs`: Implements direct parsing logic for UI-specific needs (metadata and previews).
        - `safetensors` crate: Used directly by `routes.rs`.
        - `serde_json` crate: Used directly by `routes.rs`.
        - `bytemuck` crate: Used by `routes.rs` for tensor previews.
        - (Potentially other modules like `src/native/model_loader.rs` or `src/tokenizer.rs` for different, non-UI functionalities).
    - **Key Dependencies/Technologies:** `safetensors` crate, `serde_json` crate, `tokenizers` crate (for the full `TokenizerWrapper`).
    - **Interactions with UI Backend:** The `src/ui/routes.rs` module (UI Backend) currently performs its own parsing for inspection purposes by directly invoking methods from the `safetensors` and `serde_json` crates. It does not delegate these specific UI-driven parsing tasks to higher-level abstraction functions from other parts of the `src/` library space that might be tailored for model execution rather than inspection.

## 3. Data Flow Examples

### 3.1. File Upload and Initial Processing
    1. User drags and drops file(s) onto the drop zone or selects them using the file input in `index.html`.
    2. JavaScript (`handleFormSubmit` function) captures these files, creates a `FormData` object.
    3. A `fetch` POST request is made to the `/upload` endpoint on the Web UI Backend, sending the `FormData`.
    4. The Actix web server (`src/ui/routes.rs::upload_files` handler) receives the multipart stream.
    5. For each file in the stream:
        a. File bytes are read into a `Vec<u8>`.
        b. The `process_single_file` helper function is called.
        c. If `.safetensors`: uses the `safetensors` crate to deserialize and extract metadata (tensor names, shapes, dtypes) and generate a data preview. These are stored as structured strings (e.g., "tensor_name.shape", "tensor_name.preview").
        d. If `tokenizer.json`: uses the `serde_json` crate to parse, and a preview of the JSON content is generated.
        e. If unsupported: a "skipped" message is generated.
        f. Errors during parsing are caught and formatted.
    6. The `upload_files` handler aggregates results and errors, then generates an HTML fragment. This HTML includes lists of tensors with data attributes for details and checkboxes for comparison.
    7. The JavaScript `fetch` promise resolves, and the received HTML text is injected into the `#uploadResults` div in `index.html`.
    8. JavaScript then calls `setupTensorInteraction()` to attach click listeners to tensor items (for detail view) and checkboxes (for comparison selection).

### 3.2. Detailed Tensor View
    1. User clicks on a tensor list item (`<li>`) in the UI (`index.html`) after files have been processed.
    2. The JavaScript event listener attached by `attachTensorClickListeners()` on the `<li>` element is triggered.
    3. The handler extracts the tensor name (from text content) and retrieves `data-shape`, `data-dtype`, and `data-preview` attributes from the clicked `<li>`.
    4. The `showTensorDetails(name, shape, dtype, preview)` JavaScript function is called.
    5. This function constructs HTML content displaying these details and sets it as the `innerHTML` of the `#tensorModalContent` div.
    6. The `#tensorModal` is made visible by setting its `style.display` to `block`.

### 3.3. Tensor Comparison
    1. User selects exactly two tensor checkboxes (`<input type="checkbox" name="tensor_compare_select">`) in `index.html`.
    2. The `updateCompareButtonState()` JavaScript function (triggered by checkbox `change` events) enables the "Compare Selected Tensors" button.
    3. User clicks the "Compare Selected Tensors" button (`#compareTensorsButton`).
    4. The button's JavaScript event listener:
        a. Retrieves the two checked checkboxes.
        b. For each checkbox, it uses `getTensorDataFromCheckbox()` to find the parent `<li>` and read its `data-tensor-id`, `data-shape`, `data-dtype`, and `data-preview` attributes. Tensor name and filename are parsed from `data-tensor-id`. Preview string is parsed into a numeric array.
        c. Populates the `#tensorAContent` and `#tensorBContent` divs in the `#comparisonModal` with the details (Name, File, Shape, DType, Preview) of the two selected tensors.
        d. Compares shapes and dtypes:
            i. If different, an incompatibility message is generated for `#analysisContent`.
            ii. If compatible, a compatibility message is shown, and an element-wise numerical difference of the preview arrays is calculated and formatted.
        e. The generated analysis HTML is set as the `innerHTML` of `#analysisContent`.
    5. The `#comparisonModal` is made visible (`style.display = 'block'`).

## 4. Design Choices & Patterns
    - **REST-like API for UI:** A single `/upload` endpoint is used for file processing, simplifying the backend API surface for the current UI needs.
    - **Server-Side HTML Generation for Initial Results:** The backend constructs and returns HTML fragments for the list of processed files and errors. This simplifies the frontend's initial rendering task, as it directly injects this HTML.
    - **Client-Side Enhancement for Modals & Interactions:** Detailed views (single tensor, comparison) are handled client-side using JavaScript. Data for these views is primarily read from HTML `data-` attributes generated by the server, minimizing additional backend calls for these specific interactions.
    - **Vanilla JavaScript:** The entire frontend logic is implemented using standard browser JavaScript (ES6+), without external frontend frameworks or libraries, keeping it lightweight.
    - **Pico.CSS:** A minimalist CSS framework is used for base styling, providing a clean and modern appearance with relatively few custom CSS overrides.
    - **Direct Crate Usage in UI Backend:** For UI-specific inspection tasks (like generating previews or extracting specific metadata), the UI backend (`src/ui/routes.rs`) directly utilizes crates like `safetensors`, `serde_json`, and `bytemuck`. This provides flexibility for UI needs but means it doesn't always go through more abstract data loading or processing functions that might exist in other parts of the `src/` library (e.g., those tailored for full model inference).
    - **Progressive Enhancement:** Basic HTML structure is served, and JavaScript enhances it with drag-and-drop, dynamic updates, and modal interactions.

## 5. Extensibility
    - **UI Backend:** New API routes and handlers can be added within `src/ui/routes.rs` to support more complex interactions or data processing tasks if required by the frontend. The modular structure of Actix facilitates this.
    - **Frontend:** The JavaScript code is organized into functions for specific tasks (e.g., `handleFormSubmit`, `showTensorDetails`, `setupTensorInteraction`). This allows for new UI components or interactions to be added with reasonable isolation. New modals or display areas can be added to `index.html` and controlled via JavaScript.
    - **Core Logic:** The broader Rust library in `src/` (including `src/native/` if it were more integrated) is designed for core model operations. While the UI currently performs direct parsing for inspection, it could be refactored in the future to call more abstracted functions from the core library if those functions evolve to support the UI's specific data extraction needs without the overhead of full model loading.
    - **Styling:** CSS is primarily managed by Pico.CSS, with specific custom styles for UI elements like modals and the drop zone. New components can leverage Pico.CSS classes or have specific custom styles added.
    - **File Type Support:** The `process_single_file` function in `routes.rs` can be extended with more `else if` blocks to handle new file types and their specific parsing and data extraction logic. The frontend would then need corresponding updates to display this new information.
