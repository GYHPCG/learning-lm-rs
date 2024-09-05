use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

// pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
//     // todo!("实现 rms_norm，计算前做一些必要的检查会帮助你后续调试")
//     let len = y.size();
//     assert!(len == x.size());
    
//     // x,y张量的每一个行向量和w向量的长度应该一致
//     let vector_size = w.size();
//     assert!(vector_size * x.shape()[0] == len);

//     let _x = x.data();
//     let _y = unsafe { y.data_mut() };
//     let _w = w.data();

//     for i in 0..x.shape()[0] {
//         let mut sum_squares = 0.0;
//         // 遍历向量的每一个元素
//         for j in 0..vector_size {
//             let idx = i * vector_size + j;
//             sum_squares += _x[idx] * _x[idx];
//         }

//         // 求出分母
//         let rms = (sum_squares / vector_size as f32 + epsilon).sqrt();

//         for j in 0..vector_size {
//             let idx = i * vector_size + j;
//             _y[idx] = (_x[idx] * _w[j]) / rms;
//         }
//     }

// }

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape = y.shape().clone();
    let len = shape.len();
    let last_dim = len - 1;
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();

    let mut ext_loop = 1;
    for i in 0..(shape.len() - 1) {
        ext_loop *= shape[i];
    }
    let inner_size = shape[last_dim];

    for i in 0..ext_loop {
        let mut xp = 0f32;
        for j in 0..shape[last_dim] {
            xp += _x[i * inner_size + j] * _x[i * inner_size + j];
            _y[i * inner_size + j] = _w[j] * _x[i * inner_size + j];
        }
        xp = f32::sqrt(xp / inner_size as f32 + epsilon);
        for j in 0..shape[last_dim] {
            _y[i * inner_size + j] /= xp;
        }   
    }
}

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    // let len = y.size();
    // assert!(len == x.size());

    // let _y = unsafe { y.data_mut() };
    // let _x = x.data();

    // todo!("实现 silu，这里给了一些前期准备工作的提示，你可以参考")
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    // silu(y) = silu(x) * y
    // silu(x) = sigmoid(x) * x
    for i in 0..len {
        let sigmoid = 1.0 / (1.0 + (-_x[i]).exp());
        _y[i] = (_x[i] * sigmoid) * _y[i];
    }
}

pub fn silu1(y: &mut Tensor<f32>) {
    let len = y.size();
    let _y = unsafe { y.data_mut() };
    // silu(y) = silu(x) * y
    // silu(x) = sigmoid(x) * x
    for i in 0..len {
        let sigmoid = 1.0 / (1.0 + (-_y[i]).exp());
        _y[i] = _y[i] * sigmoid;
    }
}
// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
// pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
//     let a_data = a.data(); // 获取 A 的数据
//     let b_data = b.data(); // 获取 B 的数据
//     let c_shape = c.shape().clone();
//     let c_data = unsafe { c.data_mut() }; // 获取 C 的可变引用

//     let a_shape = a.shape();
//     let b_shape = b.shape();
//     // 检查矩阵形状是否匹配
//     assert_eq!(a_shape[1], b_shape[1], "A's columns must match B's columns when B is transposed");
//     assert_eq!(c_shape[0], a_shape[0], "C's rows must match A's rows");
//     assert_eq!(c_shape[1], b_shape[0], "C's columns must match B's rows when B is transposed");

//     let a_rows = a_shape[0];
//     let a_cols = a_shape[1];
//     let b_rows = b_shape[0]; // 实际上是 B 的列数,转置
//     let c_cols = c_shape[1];

//     // 遍历 C 的每一个元素
//     for i in 0..a_rows {
//         for j in 0..b_rows {
//             // 计算 A 的第 i 行和 B^T 的第 j 行的点积
//             let mut sum = 0.0;
//             for k in 0..a_cols {
//                 sum += a_data[i * a_cols + k] * b_data[j * a_cols + k];
//             }
//             // 更新 C 矩阵
//             c_data[i * c_cols + j] = beta * c_data[i * c_cols + j] + alpha * sum;
//         }
//     }
// }
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let a_data = a.data(); // 获取 A 的数据
    let b_data = b.data(); // 获取 B 的数据
    let c_shape = c.shape().clone();
    let c_data = unsafe { c.data_mut() }; // 获取 C 的可变引用

    let a_shape = a.shape();
    let b_shape = b.shape();

    // 确定 a 和 b 的形状，并处理广播
    let a_rows = a_shape[a_shape.len() - 2];
    let a_cols = a_shape[a_shape.len() - 1];
    let b_rows = b_shape[b_shape.len() - 2];
    let b_cols = b_shape[b_shape.len() - 1];

    assert_eq!(a_cols, b_cols, "A's columns must match B's columns when B is transposed");
    assert_eq!(c_shape[c_shape.len() - 2], a_rows, "C's rows must match A's rows");
    assert_eq!(c_shape[c_shape.len() - 1], b_rows, "C's columns must match B's rows when B is transposed");

    // 处理批次维度
    let batch_size = a_shape[..a_shape.len() - 2]
        .iter()
        .zip(b_shape[..b_shape.len() - 2].iter())
        .map(|(a_dim, b_dim)| a_dim.max(b_dim))
        .product();

    // 遍历批次
    for batch_idx in 0..batch_size {
        let a_offset = batch_idx * a_rows * a_cols;
        let b_offset = batch_idx * b_rows * b_cols;
        let c_offset = batch_idx * a_rows * b_rows;

        // 遍历 C 的每一个元素
        for i in 0..a_rows {
            for j in 0..b_rows {
                // 计算 A 的第 i 行和 B^T 的第 j 行的点积
                let mut sum = 0.0;
                for k in 0..a_cols {
                    sum += a_data[a_offset + i * a_cols + k] * b_data[b_offset + j * b_cols + k];
                }
                // 更新 C 矩阵
                c_data[c_offset + i * b_rows + j] = beta * c_data[c_offset + i * b_rows + j] + alpha * sum;
            }
        }
    }
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
