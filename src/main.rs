mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    println!("Test: {}", input);
    let binding = tokenizer.encode(input, true).unwrap();
    println!("binding over");
    let input_ids = binding.get_ids();
    println!("input_ids over");
    print!("\n{}\n", input);
    println!("input_ids: {:?}",input_ids);
    let output_ids = llama.generate(
        input_ids,
        500,
        0.9,
        4,
        1.,
    );
    // println!("output_ids over");
    // println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}
