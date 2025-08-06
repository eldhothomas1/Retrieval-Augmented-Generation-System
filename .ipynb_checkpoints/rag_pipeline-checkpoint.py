from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set model ID
model_id = "tiiuae/falcon-7b-instruct"

# Load tokenizer and fix pad token
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Avoid pad token warning

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Run test prompt
prompt = "What is the capital of France?"
outputs = generator(prompt, max_new_tokens=50, do_sample=True)

# Print output
print("Output:")
print(outputs[0]["generated_text"])
