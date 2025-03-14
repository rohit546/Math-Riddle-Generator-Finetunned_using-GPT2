import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_gpt2_math_riddles"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Move model to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit UI
st.title("Math Riddle Solver")
st.write("Ask a math riddle, and the AI will try to answer it!")

# User input
user_input = st.text_input("Enter your riddle:", "What number becomes zero when you subtract 15 from half of it?")

if st.button("Solve Riddle"):
    with st.spinner("Thinking..."):
        input_text = f"Riddle: {user_input} Solution:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate response
        output_ids = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the answer
        solution = response.split("Solution:")[-1].strip()
        st.success(f"Solution: {solution}")
