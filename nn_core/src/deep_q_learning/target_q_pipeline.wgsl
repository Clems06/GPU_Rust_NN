fn get_finished(i: u32) -> bool {
    let word = finished_bits[i >> 5u];
    let mask = 1u << (i & 31u);
    return (word & mask) != 0u;
}

struct Params {
    batch_size: u32,
    layer_size: u32,

    _pad0: u32,
    _pad1: u32,

};

@group(0) @binding(0)
var<storage, read_write> current_q_values: array<f32>;

@group(0) @binding(1)
var<storage, read_write> target_q_values: array<f32>;

@group(0) @binding(2)
var<storage, read> finished_bits: array<u32>;

@group(0) @binding(3)
var<storage, read> rewards: array<f32>;

@group(0) @binding(4)
var<storage, read> actions: array<u32>;

@group(0) @binding(5)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.batch_size) { return; }

    let gamma = 0.99;
    var target_value= rewards[row];

    let base = row * params.layer_size;
    let act = actions[row];

    let curr = current_q_values[base + act];

    if !get_finished(row) {
        var max_val = -1.0e38;
        for (var i = 0u; i < params.layer_size; i += 1) {
            let idx = base + i;
            max_val = max(max_val, target_q_values[idx]);
        }
        target_value += gamma * max_val;
    } 


    for (var j = 0u; j < params.layer_size; j = j + 1u) {
        current_q_values[base + j] = 0.0;
    }


    current_q_values[base + act] =  (curr - target_value);




}