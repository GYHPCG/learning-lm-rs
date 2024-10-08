use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, masked_softmax, matmul_transb, rms_norm, silu, silu1};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();

        // 打印出safetensors中所有张量的名字
        // println!("safetensors tensors: ------------");
        // for (name, _) in safetensor.tensors() {
        //     println!("{}", name);
        // }
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        // println!("embedding is ok");
        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            // println!("rms_norm is ok");

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            
            // println!("q is ok");

            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            
            // println!("k is ok");
            
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            
            // println!("matmul_transb is ok");

            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            
            // println!("rope is ok");

            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            // readme的self-attention
            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            // println!("self_attention start");

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            // println!("self_attention end");
            // 5. Down projection and residual connection
            let mut down_proj_buf = Tensor::<f32>::default(&vec![seq_len, self.d]);
            OP::matmul_transb(
                &mut down_proj_buf,
                0.,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );
            //   residual = residual.add(&down_proj_buf); // Add residual connection
            let res_size = residual.size();
            let res_data = unsafe { residual.data_mut() };
            let down_proj_data = down_proj_buf.data();
            for i in 0..res_size {
                res_data[i] += down_proj_data[i];
            }

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        
        let mut result = token_ids.to_vec();
        let mut input_tensor = Tensor::new(token_ids.to_vec(), &vec![1, token_ids.len()]);
        // let mut cache = KVCache::new(
        //     // self.n_layers,
        //     // max_len,
        //     // self.n_kv_h * self.dqkv,
        //     // self.max_seq_len,
        // ); // 初始化缓存
        let mut cache = self.new_cache();
        // println!("cache is ok");

        for _ in 0..max_len {
            let logits = self.forward(&input_tensor, &mut cache);
            // print!("logits is ok");
            // 使用 temperature, top_k, top_p 进行采样
            let next_token = OP::random_sample(&logits, top_k as f32, top_p as u32, temperature);
            
             // 假如 next_token 是结束标记，则退出循环
             if next_token == self.eos_token_id {
                break;
            }

            result.push(next_token);

            // 更新输入张量，仅使用最新生成的 token
            input_tensor = Tensor::new(vec![next_token], &vec![1]);

           
        }
        result
    }
}



// fn self_attention(
//     hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
//     att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
//     q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
//     k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
//     n_kv_h: usize,
//     n_groups: usize,
//     seq_len: usize,
//     total_seq_len: usize,
//     dqkv: usize,
// ) {
//     for g in 0..n_groups {
//         for h in 0..n_kv_h {
//             let q_idx_start = g * n_kv_h * dqkv + h * dqkv;
//             let k_idx_start = h * dqkv;

//             // 获取 q 和 k 的子张量
//             let q_sub = q.slice(q_idx_start, &vec![seq_len, dqkv]);
//             let k_sub = k.slice(k_idx_start, &vec![total_seq_len, dqkv]);

//             // 计算 Q 和 K 的点积，存储到 att_scores 中
//             let mut att_scores_sub = Tensor::new(
//                 vec![0.0; seq_len * total_seq_len],
//                 &vec![seq_len, total_seq_len],
//             );
//             matmul_transb(&mut att_scores_sub, 0.0, &q_sub, &k_sub, 1.0);

//             // 将计算结果拷贝到 att_scores 中
//             let att_scores_data = unsafe { att_scores.data_mut() };
//             let att_scores_sub_data = att_scores_sub.data();

//             for i in 0..seq_len {
//                 for j in 0..total_seq_len {
//                     att_scores_data[h * seq_len * total_seq_len + i * total_seq_len + j] =
//                         att_scores_sub_data[i * total_seq_len + j];
//                 }
//             }
//         }
//     }

//     // 2. 对注意力分数应用 Masked Softmax
//     masked_softmax(att_scores);

//     // 3. 计算上下文向量
//     for g in 0..n_groups {
//         for h in 0..n_kv_h {
//             let v_idx_start = h * dqkv;

//             // 获取 att_scores 和 v 的子张量
//             let att_scores_sub =
//                 att_scores.slice(h * seq_len * total_seq_len, &vec![seq_len, total_seq_len]);
//             let v_sub = v.slice(v_idx_start, &vec![total_seq_len, dqkv]);

//             // 计算 att_scores 和 v 的矩阵乘积，存储到 hidden_states 中
//             let mut hidden_states_sub =
//                 Tensor::new(vec![0.0; seq_len * dqkv], &vec![seq_len, dqkv]);
//             matmul_transb(&mut hidden_states_sub, 0.0, &att_scores_sub, &v_sub, 1.0);

//             // 将计算结果拷贝到 hidden_states 中
//             let hidden_states_data = unsafe { hidden_states.data_mut() };
//             let hidden_states_sub_data = hidden_states_sub.data();

//             for i in 0..seq_len {
//                 for j in 0..dqkv {
//                     hidden_states_data[g * n_kv_h * dqkv + h * dqkv + i * dqkv + j] =
//                         hidden_states_sub_data[i * dqkv + j];
//                 }
//             }
//         }
//     }
// }
fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // println!("{:?}, {:?}", hidden_states.shape(), att_scores.shape());
    // println!("{:?}, {:?}", q.shape(), k.shape());
    // println!("{:?}", v.shape());
    // println!("{}, {}, {}, {}, {}", n_kv_h, n_groups, seq_len, total_seq_len, dqkv);
    let dim = dqkv;
    let seq_dim = n_kv_h * dqkv;
    let hidden_len = n_kv_h * n_groups * dqkv;
    let hidden_data = unsafe {
        hidden_states.data_mut()
    };

    let att_dim_3 = total_seq_len;
    let att_dim_2 = seq_len * total_seq_len;
    let att_dim_1 = n_groups * att_dim_2;
    let att_ptr = unsafe {
        att_scores.data_mut()
    };
    for x in 0..seq_len {
        for y in 0..total_seq_len {
            for i in 0..n_kv_h {
                for group in 0..n_groups {
                    let start_q = (i * n_groups + group) * dim + seq_dim * n_groups * x;
                    let q_vec = &q.slice(start_q, &vec![16, 1]);
                    let start_k = i * dim + seq_dim * y;
                    let k_vec = &k.slice(start_k, &vec![16, 1]);
                    let value = OP::dot(q_vec, k_vec) / f32::sqrt(dim as f32);
                    // assert!(i * att_dim_1 + group * att_dim_2 + x * att_dim_3 + y < n_kv_h * n_groups * seq_len * total_seq_len);
                    att_ptr[i * att_dim_1 + group * att_dim_2 + x * att_dim_3 + y] = value; 
                }
            }
        }
    }
    masked_softmax(att_scores);
    let v_ptr = v.data();
    for i in 0..n_kv_h  {
        for g in 0..n_groups {
            let att_start = att_dim_1 * i + g * att_dim_2;
            let att_mat = &att_scores.slice(att_start, &vec![seq_len, total_seq_len]);
            let mut data = vec![0f32; dqkv * total_seq_len];
            for row in 0..dqkv {
                let d_start = row * total_seq_len;
                for col in 0..total_seq_len {
                    data[d_start + col] = v_ptr[col * dqkv * n_kv_h + i * dqkv + row];
                }
            }
            let v_mat: Tensor<f32> = Tensor::new(data, &vec![dqkv, total_seq_len]);
            let mut t_mat: Tensor<f32> = Tensor::default(&vec![seq_len, dqkv]);
            OP::matmul_transb(&mut t_mat, 0f32, att_mat, &v_mat, 1f32);
            let t_data = t_mat.data();
            for row in 0..seq_len {
                for col in 0..dqkv {
                    let hidden_p = row * hidden_len + (i * n_groups + g) * dqkv + col;
                    hidden_data[hidden_p] = t_data[row * dqkv + col];
                }
            }
        }
    }
    // todo!("Implement self_attention");
}


fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0.0, hidden_states, w_gate, 1.0);
    matmul_transb(up, 0.0, hidden_states, w_up, 1.0);
    silu(up, gate);
    matmul_transb(hidden_states, 1.0, up, w_down, 1.0);

    let res_size = residual.size();
    let res_data = unsafe { residual.data_mut() };
    let hid_data = hidden_states.data();
    for i in 0..res_size {
        res_data[i] += hid_data[i];
    }
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
