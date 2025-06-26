model_names = ["openai-community/gpt2-large",
               "openai-community/gpt2-xl",
               "facebook/opt-1.3b",
               "tiiuae/Falcon3-1B-Instruct"]
pretrained_model = model_names[0]

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForCausalLM.from_pretrained(pretrained_model).to(device)

import torchinfo
summary = torchinfo.summary(model, 
                  input_data=torch.randint(0, tokenizer.vocab_size, (32, 64), device=device),
                  col_names=["input_size", "output_size", "num_params"])
print(summary)

# Define tasks and example inputs
tasks = {
    "story completion": "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley in the Andes Mountains.",
    "question answering": "Q: Who wrote the book the origin of species? A:",
    "translation": "Translate English to French: I'm home = Je suis là | I love you = ",
    "reading comprehension": "Tom goes everywhere with Catherine Green, a 54-year-old secretary. Catherine and Tom live in Sweden. Q: How old is Catherine? A: 54 | Q: Where does she live? A:"
}

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=64,
            do_sample=True,
            num_beams=4,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Run each task
for task, input_text in tasks.items():
    print(f"\n--- {task.upper()} ---")
    print("Input:", input_text)
    output = generate(input_text)
    # Remove input prefix from the output
    output = output[len(input_text):].strip()
    print("Output:", output)
