use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // safetensors里存储的是原始数据，你需要以FP32的形式读取出来，创建出项目所使用的张量。
        //safetensors包含张量的形状，你无需对原始张量做任何变形
        let get_tensor = |name: &str| {
            // 查看safetensor源码获取
            let tensor = safetensor.tensor(name).unwrap();
            let data = tensor.data();
            let shape = tensor.shape();
    
            let f32_data: Vec<f32> = data.chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
            let shape_vec: Vec<usize> = shape.to_vec();

            Tensor::new (f32_data, &shape_vec)

        };
        
         // Helper function to get multiple tensors from safetensor for each layer
         let get_layer_tensors = |suffix: &str, layers: usize| {
            (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.{}", i, suffix)))
                .collect()
        };

        let num_layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: get_layer_tensors("input_layernorm.weight", num_layers),
            wq: get_layer_tensors("self_attn.q_proj.weight", num_layers),
            wk: get_layer_tensors("self_attn.k_proj.weight", num_layers),
            wv: get_layer_tensors("self_attn.v_proj.weight", num_layers),
            wo: get_layer_tensors("self_attn.o_proj.weight", num_layers),
            rms_ffn_w: get_layer_tensors("post_attention_layernorm.weight", num_layers),
            w_up: get_layer_tensors("mlp.up_proj.weight", num_layers),
            w_gate: get_layer_tensors("mlp.gate_proj.weight", num_layers),
            w_down: get_layer_tensors("mlp.down_proj.weight", num_layers),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
