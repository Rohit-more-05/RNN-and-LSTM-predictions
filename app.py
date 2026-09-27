import streamlit as st
from model_utils import load_resources, predict_next_words
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered"
)

# --- Custom CSS for better aesthetics ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced4da;
    }
    .stButton button {
        border-radius: 8px;
        background-color: #007bff;
        color: white;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0056b3;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-box {
        padding: 20px;
        border-radius: 8px;
        background-color: #e9ecef;
        border-left: 5px solid #007bff;
        margin-top: 20px;
        font-size: 1.1em;
        line-height: 1.5;
    }
    .info-text {
        font-size: 0.9em;
        color: #6c757d;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🔮 Next Word & Sentence Predictor")
st.markdown('<p class="info-text">Powered by Deep Learning (LSTM & RNN)</p>', unsafe_allow_html=True)

# --- Load Resources ---
@st.cache_resource
def get_models_and_tokenizer():
    # Pass the directory where the current script is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    return load_resources(base_path)

try:
    with st.spinner('Loading models and tokenizer... Please wait.'):
        model_lstm, model_rnn, tokenizer, max_len = get_models_and_tokenizer()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

# --- Main Interface ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Configuration")
    
    # Model Selection Bifurcation
    st.markdown("**Choose Prediction Engine:**")
    model_choice = st.radio(
        "Select Model",
        options=["LSTM Model", "RNN Model"],
        help="LSTM (Long Short-Term Memory) generally performs better on longer text sequences compared to standard RNNs.",
        label_visibility="collapsed"
    )
    
    # Task Type Selection (Word vs Sentence)
    st.markdown("**Choose Task:**")
    task_choice = st.radio(
        "Task Type",
        options=["Next Word Prediction", "Next Sentence (Multiple Words)"],
        label_visibility="collapsed"
    )

    if task_choice == "Next Sentence (Multiple Words)":
        num_words_to_predict = st.slider("Number of words to generate:", min_value=2, max_value=20, value=5)
    else:
        num_words_to_predict = 1

with col2:
    st.subheader("✍️ Input Text")
    user_input = st.text_area(
        "Enter your starting phrase:",
        placeholder="Type something here...",
        height=150
    )
    
    # Select the corresponding model
    active_model = model_lstm if model_choice == "LSTM Model" else model_rnn
    
    generate_btn = st.button("Generate Prediction ✨")

# --- Prediction Logic ---
if generate_btn:
    if not user_input.strip():
        st.warning("Please enter some text to predict the next word(s).")
    else:
        with st.spinner("Generating..."):
            try:
                # Predict
                prediction_result = predict_next_words(
                    model=active_model,
                    tokenizer=tokenizer,
                    max_len=max_len,
                    text=user_input.strip(),
                    num_words=num_words_to_predict
                )
                
                # Highlight the generated part
                # The prediction_result contains both the input and the new words.
                input_length = len(user_input.strip())
                original_text = prediction_result[:input_length]
                generated_text = prediction_result[input_length:]
                
                st.subheader("🎯 Result")
                st.markdown(
                    f'<div class="result-box">'
                    f'<span>{original_text}</span>'
                    f'<span style="color: #007bff; font-weight: bold;">{generated_text}</span>'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# --- Footer or extra info ---
st.markdown("---")
st.markdown("""
<small>
**Model Metrics Info:** Detailed validation loss/accuracy history is typically not preserved when models are saved to `.h5` files without their `history` objects. Therefore, validation metrics are not displayed here, but both models have been optimized for sequential text prediction.
</small>
""", unsafe_allow_html=True)
