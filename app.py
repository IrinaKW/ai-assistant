import streamlit as st
from openai import OpenAI
import base64
import os

st.set_page_config(
    page_title="Irina White | AI Assistant",
    page_icon="🤖",
    layout="centered"
)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@st.cache_data(show_spinner=False)
def get_resume_text():
    resume_text = ""
    images = ["data/resume_page1.png", "data/resume_page2.png"]
    
    for img_path in images:
        if os.path.exists(img_path):
            base64_image = encode_image(img_path)
            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this resume page."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                }]
            )
            resume_text += response.choices[0].message.content + "\n\n"
    return resume_text

with st.spinner("Initializing assistant..."):
    resume_text = get_resume_text()

SYSTEM_PROMPT = f"""
You are the professional AI Assistant for Irina White, an AI Engineer & Data Scientist.
Use the following resume text to answer questions:
{resume_text}

Rules:
1. Answer professionally based ONLY on the provided resume.
2. Keep answers concise.
3. If information is missing, suggest contacting her via LinkedIn.
"""

st.title("Chat with Irina's AI Assistant")
st.markdown("*Powered by Llama 3.2 Vision & Llama 3.3 Text on Groq.*")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! What would you like to know about Irina's background?"}
    ]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask about Irina's experience or projects..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=st.session_state.messages
        )
        full_response = response.choices[0].message.content
        st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
