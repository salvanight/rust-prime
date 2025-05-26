## 3. Ergonomic and User-Friendly API (Broadcasting, Slicing, Traits, etc.)

A high-level, ergonomic API will significantly increase the adoption of the tensor engine. It's recommended to offer functionality similar to that of NumPy or PyTorch in terms of convenience:

### 3.1. Automatic Broadcasting
Allow operations between tensors of different but compatible shapes to perform implicit broadcasting.
*   **Technical Term**: "Broadcasting" is a set of rules for applying binary operations on arrays of different shapes. It implies that if a tensor has dimensions of size 1 or lacks a dimension compared to another tensor in an operation, its values are virtually repeated along that dimension to match the larger tensor's shape.
*   For example, adding a tensor of shape (3, 1) (a column vector) to another of shape (3, 4) should result in a (3, 4) tensor, where the column vector is added to each column of the second tensor.
*   Internally, implementing the `std::ops::Add` trait for `Tensor<T>` can handle this logic: check if shapes differ in length or if any axis has size 1, and iterate appropriately during the operation.
*   **Examples**:
    *   The `l2` library (inspired by PyTorch) supports broadcasting and most mathematical operations naturally ([github.com](https://github.com/LaurentMazare/l2)).
    *   `RSTSR` also highlights full support for broadcasting and n-dimensional operations ([github.com](https://github.com/mratsim/RSTSR)).
*   For the end-user, this allows writing `let c = &a + &b;` without manually ensuring dimensions are identical.

### 3.2. NumPy-style Slicing
Offer easy ways to extract subsets of data (sub-tensors) ideally without copying data, known as "views." In Rust, the `[]` operator cannot be directly overloaded for multiple, variadic indices as in Python, but several patterns can be employed:

    #### 3.2.1. Indexing with `Index` and `IndexMut`
    Implement the `std::ops::Index` and `std::ops::IndexMut` traits for your `Tensor` type to accept tuples as indices (e.g., `impl Index<(usize, usize)> for Tensor<T>` for 2D tensors), allowing syntax like `tensor[(i, j)]`.
    *   **Technical Term**: `Index` and `IndexMut` are standard Rust traits for overloading the indexing operators (`[]`).
    *   For N dimensions, you could make them accept `Index<&[usize]>` or provide methods like `.get(&[i, j, k])`.

    #### 3.2.2. Slice Methods and Views (`TensorView`)
    Provide a `slice` method, or even a macro similar to `ndarray`'s `s![]` macro (e.g., `tensor.slice(s![0..10, ..])`).
    *   A simple implementation could accept ranges as parameters: `fn slice(&self, ranges: &[Range<usize>]) -> TensorView<T>`. This method would calculate the appropriate data offsets and strides and return a "view" (a reference or a new struct that refers to the original data but doesn't own it).
    *   **Technical Term**: A "view" (often called `TensorView` or `ArrayView`) is a tensor-like object that refers to data owned by another tensor. Operations on views often directly manipulate the original tensor's data. This is crucial for efficiency, as it avoids data copies, a common practice in NumPy and `ndarray` ([docs.rs/ndarray](https://docs.rs/ndarray/latest/ndarray/struct.ArrayView.html)).
    *   Ensure your `Tensor` design can represent a view, either by having a flag indicating ownership vs. view status, or by having a separate `TensorView` type.

    #### 3.2.3. Fancy Indexing (Future Consideration)
    Support for boolean slicing (indexing with a boolean tensor) or slicing with lists of indices ("fancy indexing") is also useful, though basic range slicing will cover most initial use cases.

### 3.3. Flexible Reshaping and Dimension Manipulation
Include methods to fluently reconfigure a tensor's dimensions.
*   `tensor.reshape(&[new_dims])`: Should return a new `Tensor` (or `TensorView`) that shares the same data but interprets it with a new shape (after validating that the total number of elements matches). An in-place reshape is possible by updating metadata if no new object allocation is desired.
*   `tensor.expand_dims(axis)`: To add new dimensions of size 1.
*   `tensor.squeeze()`: To remove dimensions of size 1.
*   These operations make the API more user-friendly by avoiding manual shape manipulation.

### 3.4. Operator Overloading for Arithmetic (`std::ops`)
Implement traits from `std::ops` (e.g., `Add`, `Sub`, `Mul`, `Div`) for `Tensor` to allow natural syntax like `a + b`, `-tensor`, or `tensor1 * tensor2`.
*   The semantics (element-wise or linear algebra) should be clearly defined.
    *   In NumPy, `*` is element-wise ([docs.rs/numpy](https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)).
    *   For matrix multiplication, NumPy uses a dedicated method (`.dot()`) or the `@` operator in Python.
*   In your design, you might decide that `Tensor * Tensor` is element-wise (as `ndarray` does: arithmetic operators work element-wise - [docs.rs/ndarray](https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#arithmetic-operations)).
*   For matrix products, provide an explicit method like `matmul(&self, &other)` or consider overloading a different operator.
    *   **Example**: `RSTSR` uses the `%` operator for matrix multiplication ([github.com](https://github.com/mratsim/RSTSR)), as `%` was otherwise unused for its tensor types. This is an optional but interesting choice for readability.
*   Operator overloading in Rust is achieved by implementing the corresponding trait from `std::ops` ([doc.rust-lang.org](https://doc.rust-lang.org/std/ops/index.html)).

### 3.5. Integration with Standard Rust Traits
Beyond `Index` and arithmetic operators:
*   Implement `Debug` and `Display` for legible tensor printing (e.g., similar to NumPy's output).
*   Implement `IntoIterator` to allow iteration over elements (perhaps yielding references or values).
*   If applicable, implement or use traits from community crates, e.g., `ndarray::IntoDimension` or conversions like `AsRef<[T]>` if the tensor is 1D.
*   The goal is for the tensor to behave as much like a native Rust collection as possible.

### 3.6. API Consistency and Safety
The API must validate preconditions.
*   If tensors with incompatible shapes (and not broadcastable) are added, return a clear error (e.g., `Err(TensorError::ShapeMismatch)`) or `panic!` with a descriptive message.
*   Similarly, for indexing: if an index is out of bounds for any dimension, it's better to return an error than to cause an illegal memory access (though Rust's memory safety prevents segfaults, logical invariants are your responsibility).
*   You can draw inspiration from `ndarray`'s error handling (e.g., it throws a `ShapeError` for dimension condition failures).

In summary, the aim is to make using tensors in Rust as close as possible to the experience of using NumPy: easy tensor creation, concise indexing and slicing, natural mathematical operations with operators, and effortless shape manipulation. Existing projects like `l2` ([github.com](https://github.com/LaurentMazare/l2)) demonstrate this is achievable, implementing NumPy-style slicing, broadcasting, and many important mathematical operations for a PyTorch-like experience. `ndarray` also provides an idiomatic Rust API that can be emulated in many aspects ([docs.rs/ndarray](https://docs.rs/ndarray/)).
