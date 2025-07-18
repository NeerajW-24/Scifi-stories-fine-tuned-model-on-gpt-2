import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

@st.cache_resource
def load_generator():
    model_id = "Neeraj24/gpt2-scifi-stories-model"  # Your Hugging Face repo
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_generator()

st.title("🚀 Fine-Tuned GPT-2 scifi stories Generator 🌌")
st.markdown("Write scifi stories using your fine-tuned GPT-2 model....")

prompt = st.text_area("Enter a news prompt:", height=150, placeholder="e.g. The rocket touched the surface of mars....")

temperature = st.slider("Creativity (temperature)", 0.5, 1.2, 0.9, 0.1)
length = st.slider("Max output length", 50, 300, 150, 10)

if st.button("Generate"):
    with st.spinner("Generating..."):
        output = generator(prompt, max_new_tokens=length, do_sample=True, temperature=temperature)[0]["generated_text"]
        st.subheader("🛰️Generated Text")
        st.write(output)
