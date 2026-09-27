import pickle
import numpy as np
import tensorflow as tf
import os

# Cache the resources so they are loaded only once in Streamlit
def load_resources(base_path=""):
    try:
        lstm_path = os.path.join(base_path, 'history_lstm.h5')
        rnn_path = os.path.join(base_path, 'history_rnn.h5')
        tokenizer_path = os.path.join(base_path, 'tokenizer.pkl')
        max_len_path = os.path.join(base_path, 'max_len.pkl')

        # Load models
        model_lstm = tf.keras.models.load_model(lstm_path)
        model_rnn = tf.keras.models.load_model(rnn_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Load max_len
        with open(max_len_path, 'rb') as f:
            max_len = pickle.load(f)
            
        return model_lstm, model_rnn, tokenizer, max_len
    except Exception as e:
        raise Exception(f"Error loading resources: {e}")

def predict_next_words(model, tokenizer, max_len, text, num_words=1):
    """
    Predicts the next `num_words` for the given `text`.
    """
    result_text = text
    
    # Determine the expected input length for the model. 
    # Usually max_len - 1, or we can extract it from the model's input shape.
    input_shape = model.input_shape
    if input_shape and len(input_shape) > 1 and input_shape[1] is not None:
        input_len = input_shape[1]
    else:
        # Fallback to max_len - 1 which is standard for next word prediction
        input_len = max_len - 1
        
    for _ in range(num_words):
        # Tokenize the current text
        sequence = tokenizer.texts_to_sequences([result_text])[0]
        
        # Pad the sequence
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            from tensorflow.keras.utils import pad_sequences
            
        padded_sequence = pad_sequences([sequence], maxlen=input_len, padding='pre')
        
        # Predict probabilities
        predicted_probs = model.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Find the word corresponding to the predicted index
        predicted_word = None
        # tokenizer.index_word is usually a dictionary mapping index to word (faster than iterating word_index)
        if hasattr(tokenizer, 'index_word') and predicted_index in tokenizer.index_word:
            predicted_word = tokenizer.index_word[predicted_index]
        else:
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    predicted_word = word
                    break
                    
        if not predicted_word:
            break
            
        result_text += " " + predicted_word
        
    return result_text
