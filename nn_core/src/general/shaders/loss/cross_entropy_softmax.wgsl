@group(0) @binding(0) var<storage, read_write> result: array<f32>; // SoftMax probabilities and then Gradients ∂L/∂z_i
@group(0) @binding(1) var<storage, read> expected: array<f32>; // One-hot encoded targets

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;

    result[row] = result[row] - expected[row];
}