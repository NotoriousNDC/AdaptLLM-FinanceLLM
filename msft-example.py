from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AdaptLLM/finance-chat")
model = model.to('cuda')
tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/finance-chat", use_fast=True)

# Put your input here:
user_input = '''Use this fact to answer the question: Title of each class Trading Symbol(s) Name of each exchange on which registered
Common Stock, Par Value $.01 Per Share MMM New York Stock Exchange
MMM Chicago Stock Exchange, Inc.
1.500% Notes due 2026 MMM26 New York Stock Exchange
1.750% Notes due 2030 MMM30 New York Stock Exchange
1.500% Notes due 2031 MMM31 New York Stock Exchange

Which debt securities are registered to trade on a national securities exchange under 3M's name as of Q2 of 2023?'''

# We use the prompt template of LLaMA-2-Chat demo
prompt = f"<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{user_input} [/INST]"

inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
outputs = model.generate(input_ids=inputs, max_length=4096)[0]

answer_start = int(inputs.shape[-1])
pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)

print(f'### User Input:\n{user_input}\n\n### Assistant Output:\n{pred}')