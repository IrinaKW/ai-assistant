import streamlit as st
from openai import OpenAI
import base64
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Irina White | AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# 2. Connect to Groq's FREE Inference API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# 3. Vision API Setup: Encode images so the AI can "see" them
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 4. Smart OCR using Groq's Vision Model (Cached so it only runs once)
@st.cache_data(show_spinner=False)
def get_resume_text_from_images():
    resume_text = ""
    images = ["data/resume_page1.png", "data/resume_page2.png"]
    
    for img_path in images:
        if os.path.exists(img_path):
            try:
                base64_image = encode_image(img_path)
                
                # Ask the Vision model to extract the text
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct", # Groq's free Vision model
                    messages=[
                        {
                            "role": "user",
                            "content":[
                                {"type": "text", "text": "Extract all the text from this resume page exactly as it is written. Do not add any formatting or commentary, just output the raw text."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                    },
                                },
                            ],
                        }
                    ],
                )
                resume_text += response.choices[0].message.content + "\n\n"
            except Exception as e:
                return f"Error reading {img_path}: {e}"
        else:
            return f"Error: Could not find {img_path} in the folder."
            
    return resume_text

# Show a loading spinner while the Vision model reads the images on startup
with st.spinner("AI is reading Irina's latest resume images..."):
    resume_text = get_resume_text_from_images()
    
with st.expander("🔍 Debug: Click here to see what the Vision AI actually read"):
    st.write(resume_text)

# 5. The "Brain" of the Assistant
SYSTEM_PROMPT = f"""
You are the professional AI Assistant for Irina White, an AI Engineer & Data Scientist at Neurons Inc.
Irina specializes in:
- Bridging Neuroscience, Math, and Machine Learning.
- Computer Vision (YOLO, EfficientNet).
- Scalable Backend APIs (FastAPI) and bringing ML to production.
- Prompt Engineering and Multimodal LLMs.

Here is the extracted text from Irina's resume images. Use this to answer specific questions:
-----------------
{resume_text}
-----------------

Rules:
1. Answer questions about Irina confidently and professionally based ONLY on the resume provided.
2. Keep answers concise but informative. 
3. If asked something not in the resume, politely decline and suggest they contact her via LinkedIn.
"""

st.title("Chat with Irina's AI Assistant")
st.markdown("*Powered by Multimodal Llama-3 Vision & Text on Groq.*")

# 6. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages =[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "Hi! What would you like to know about Irina's background?"}
    ]

# 7. Display Chat History
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 8. Handle User Input
if prompt := st.chat_input("E.g., What are Irina's definite strengths?"):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Call Groq's super-smart Text API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Updated model name!
            messages=st.session_state.messages
        )
        
        full_response = response.choices[0].message.content
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
