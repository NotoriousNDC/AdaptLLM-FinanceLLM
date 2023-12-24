from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-LLM")
model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-LLM")
model = model.to('cuda')

# Example prompt
prompt = "Analyze the current trends in the financial market:"

# Encode the prompt
input_ids = tokenizer.encode(prompt, return_tensors='pt')
input_ids = input_ids.to('cuda')

# Generate text with specified settings in the generate method
output = model.generate(input_ids, num_return_sequences=1, no_repeat_ngram_size=2, num_beams=5, early_stopping=True)

# Decode and print the output
print(tokenizer.decode(output[0], skip_special_tokens=True))
