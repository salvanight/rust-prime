use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust_native_transformer::tensor_engine::Tensor; // Path to Tensor type
use rand::{rngs::StdRng, SeedableRng, Rng}; // For generating random tensor data

// Helper to create a random tensor (can be similar to the one in tests)
fn create_random_tensor(shape: Vec<usize>, seed: u64) -> Tensor<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let total_size = shape.iter().product();
    let data = (0..total_size).map(|_| rng.gen::<f32>()).collect::<Vec<f32>>();
    Tensor::new(data, shape).expect("Failed to create tensor for benchmark")
}

// Benchmark for matmul operations
fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMul");

    // Define a few representative sizes (M, K, N)
    let sizes = [
        (16, 32, 16), // K multiple of 8
        (16, 30, 16), // K not multiple of 8
        (64, 128, 64),
        (63, 127, 63), // Larger, non-aligned
    ];

    for &(m, k, n) in &sizes {
        let a = create_random_tensor(vec![m, k], 0);
        let b = create_random_tensor(vec![k, n], 1);

        group.bench_with_input(format!("Scalar MatMul {}x{}x{}", m, k, n), &(&a, &b), |bencher, (a_ref, b_ref)| {
            bencher.iter(|| black_box(Tensor::matmul(a_ref, b_ref).unwrap()));
        });

        group.bench_with_input(format!("SIMD MatMul {}x{}x{}", m, k, n), &(&a, &b), |bencher, (a_ref, b_ref)| {
            bencher.iter(|| black_box(Tensor::matmul(a_ref, b_ref).unwrap())); // Changed from matmul_simd
        });
    }
    group.finish();
}

// Benchmark for gelu operations
fn benchmark_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("GELU");

    let sizes = [
        1024, // Multiple of 8
        1000, // Not multiple of 8
        8192,
        8190,
    ];

    for &size in &sizes {
        // Create a 1D tensor or a 2D tensor with total size elements
        let t = create_random_tensor(vec![size], 0);

        group.bench_with_input(format!("Scalar GELU ({} elems)", size), &t, |bencher, tensor_ref| {
            bencher.iter(|| black_box(tensor_ref.gelu().unwrap()));
        });

        group.bench_with_input(format!("SIMD GELU ({} elems)", size), &t, |bencher, tensor_ref| {
            bencher.iter(|| black_box(tensor_ref.gelu().unwrap())); // Changed from gelu_simd
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_matmul, benchmark_gelu);
criterion_main!(benches);
