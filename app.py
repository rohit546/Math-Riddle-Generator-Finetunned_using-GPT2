import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

# Load fine-tuned model and tokenizer
model_path = "./fine_tuned_gpt2_math_riddles"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Move model to appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit UI
st.set_page_config(page_title="Math Riddle Solver", page_icon="🧠", layout="centered")
st.title("🧠 Math Riddle Solver 🔢")
st.write("### Test your brain with tricky math riddles! 💡")

# Display three best math riddles
best_riddles = [
    "I am a three-digit number. My tens digit is five more than my ones digit, and my hundreds digit is eight less than my tens digit. What number am I?",
    "I add five to nine and get two. The answer is correct, but how?",
    "When I take five and double it, I get ten. When I take ten and halve it, I get five. But when I take twenty and quarter it, I don't get five. Why?"
]

st.subheader("🔥 Try solving these famous math riddles:")
st.markdown(f"1️⃣ **{best_riddles[0]}**")
st.markdown(f"2️⃣ **{best_riddles[1]}**")
st.markdown(f"3️⃣ **{best_riddles[2]}**")

# User input
st.subheader("💬 Enter your own riddle below:")
user_input = st.text_input("Type your riddle here:", "What number becomes zero when you subtract 15 from half of it?")

if st.button("🔍 Solve Riddle"):
    with st.spinner("🤔 Thinking..."):
        input_text = f"Riddle: {user_input} Solution:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        # Generate response
        output_ids = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract the answer
        solution = response.split("Solution:")[-1].strip()
        
        # Display solution
        st.success(f"✅ Solution: {solution}")

# Add a cool footer
st.markdown("---")
st.markdown("💡 *Powered by AI — Bringing fun & math together!* ✨")
