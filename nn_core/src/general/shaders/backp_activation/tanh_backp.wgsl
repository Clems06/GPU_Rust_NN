struct Params {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32
};

@group(0) @binding(0) var<storage, read> dz_in: array<f32>;  // δa
@group(0) @binding(1) var<storage, read> z: array<f32>;      // z
@group(0) @binding(2) var<storage, read_write> dz_out: array<f32>; // δz
@group(0) @binding(3) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = params.batch_size * params.layer_size;
    if (idx >= total) { return; }

    let a = exp(-2*z[idx]);
    let b = (1-a)/(1+a);
    dz_out[idx] = dz_in[idx] * (1 - b*b);
}