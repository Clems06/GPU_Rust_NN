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

    // ReLU derivative
    dz_out[idx] = select(0., dz_in[idx], z[idx] > 0.0);
}