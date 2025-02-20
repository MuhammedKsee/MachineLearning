from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ve Tokenizer yükleme
model_name = "Salesforce/codegen-350M-mono"  # Python için optimize edilmiş model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Örnek Python kodu tamamlama
input_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
inputs = tokenizer(input_code, return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)

# Çıktıyı çözme ve yazdırma
completed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(completed_code)



while True:
    user_input = input("Kod Başlangıcını Girin (Çıkmak için 'exit'): ")
    if user_input.lower() == "exit":
        break
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1)
    completed_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nTamamlanmış Kod:\n", completed_code)
