struct Params {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32
};

@group(0) @binding(0) var<storage, read> a_prev: array<f32>;
@group(0) @binding(1) var<storage, read> dz: array<f32>;
@group(0) @binding(2) var<storage, read_write> dW: array<f32>;
@group(0) @binding(3) var<storage, read_write> db: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16,16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // row in W (layer_size)
    let j = gid.y; // col in W (k)

    if (i >= params.layer_size || j >= params.prev_layer_size) { return; }

    var accum: f32 = 0.0;
    var bias_accum: f32 = 0.0;

    for (var b: u32 = 0; b < params.batch_size; b++) {
        let dz_val = dz[b * params.layer_size + i];
        let a_val = a_prev[b * params.prev_layer_size + j];
        accum += dz_val * a_val;

        if (j == 0) {
            bias_accum += dz_val;
        }
    }

    dW[i * params.prev_layer_size + j] = accum;
    if (j == 0) {
        db[i] = bias_accum;
    }
}
