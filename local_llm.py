import ollama

def generate_answer(prompt):
    response = ollama.chat(
        model="qwen2.5:1.5b",
        options={
            "temperature": 0
        },
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']