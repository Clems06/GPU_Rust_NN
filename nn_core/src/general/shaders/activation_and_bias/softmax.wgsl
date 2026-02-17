struct LayerParams {
    batch_size: u32,
    layer_size: u32,
    prev_layer_size: u32,
    _pad: u32,
};

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_logsum: array<f32, 256>;

@group(0) @binding(0) var<storage, read> z_values: array<f32>;
@group(0) @binding(1) var<storage, read> biases: array<f32>;
@group(0) @binding(2) var<storage, read_write> log_activation_values: array<f32>;  // Log probabilities
@group(0) @binding(3) var<uniform> params: LayerParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let batch_index = global_id.y;
    let layer_size = params.layer_size;
    let batch_size = params.batch_size;
    
    if (batch_index >= batch_size) {
        return;
    }
    
    // Find max for numerical stability
    var max_val = -1.0e38;
    for (var i = local_id.x; i < layer_size; i += 256u) {
        let idx = batch_index * layer_size + i;
        max_val = max(max_val, z_values[idx] + biases[i % layer_size]);
    }
    
    shared_max[local_id.x] = max_val;
    workgroupBarrier();
    
    // Parallel reduction for max
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_id.x < stride) {
            shared_max[local_id.x] = max(shared_max[local_id.x], shared_max[local_id.x + stride]);
        }
        workgroupBarrier();
    }
    
    let batch_max = shared_max[0];
    
    // Compute log-sum-exp: log(∑exp(x_i - max))
    var log_sum = 0.0;
    
    for (var i = local_id.x; i < layer_size; i += 256u) {
        let idx = batch_index * layer_size + i;
        let z = z_values[idx] + biases[i % layer_size];
        log_sum += exp(z - batch_max);
    }
    
    shared_logsum[local_id.x] = log_sum;
    workgroupBarrier();
    
    // Parallel reduction for sum
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_id.x < stride) {
            shared_logsum[local_id.x] = shared_logsum[local_id.x] + shared_logsum[local_id.x + stride];
        }
        workgroupBarrier();
    }
    
    let total_sum = shared_logsum[0];
    let log_sum_exp = log(total_sum) + batch_max;
    
    // Compute log probabilities: log(exp(x_i) / ∑exp(x_j)) = x_i - log_sum_exp
    for (var i = local_id.x; i < layer_size; i += 256u) {
        let idx = batch_index * layer_size + i;
        let z = z_values[idx] + biases[i % layer_size];
        log_activation_values[idx] = z - log_sum_exp;
    }
}