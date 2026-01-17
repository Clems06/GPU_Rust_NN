struct Params {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32
};

@group(0) @binding(0)
var<storage, read> A: array<f32>;

@group(0) @binding(1)
var<storage, read> x: array<f32>;

@group(0) @binding(2)
var<storage, read_write> y: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;

    if (row >= params.batch_size * params.layer_size) {
        return;
    }

    let neuron = row % params.layer_size;
    let batch  = row / params.layer_size;

    var sum: f32 = 0.0;
    for (var k: u32 = 0; k < params.prev_layer_size; k = k + 1) {
        sum += A[neuron * params.prev_layer_size + k] * x[batch * params.prev_layer_size + k];
    }

    y[row] = sum;
}