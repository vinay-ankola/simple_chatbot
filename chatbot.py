from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []
print("Chatbot ready! (type 'exit' to quit)\n")
print("CONVERSATION HISTORY: ", conversation_history, "\n")

while True:
    # keep only last few exchanges (prevents confusion)
    conversation_history = conversation_history[-6:]

    history_string = "\n".join(conversation_history)
    print("HISTORY STRING: ", history_string, "\n")

    input_text = input("> ")
    print("INPUT TEXT: ", input_text, "\n")

    if input_text.lower() == "exit":
        break

    prompt = history_string + f"\nUser: {input_text}\nBot:"
    print("PROMPT: ", prompt, "\n")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    print("INPUTS: ", inputs, "\n")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85
    )

    print("OUTPUTS: ",outputs)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("RESPONSE: ", response, "\n")

    print("CONVERSATION HISTORY before append: ", conversation_history)
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")
    print("CONVERSATION HISTORY after append: ", conversation_history)



