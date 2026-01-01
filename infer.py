import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "diabolic6045/gemma-2-2b-chess"
LORA_PATH = "anand_gemma_lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model on CPU (NO bitsandbytes)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.float32
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def predict_next_move(moves):
    prompt = f"{moves} =>"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=6,
            temperature=0.6,
            top_k=10,
            do_sample=True
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded.split("=>")[-1].strip().split()[0]

if __name__ == "__main__":
    print(predict_next_move("e4 e5 Nf3 Nc6"))
