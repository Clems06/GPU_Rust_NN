struct Params {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> bias: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&output)) {
        let x = input[i] + bias[i%params.layer_size];
        let a = exp(-2*x);
        output[i] = (1-a)/(1+a);
    }
}