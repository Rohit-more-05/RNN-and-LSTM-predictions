import tensorflow as tf
import pickle

try:
    model_lstm = tf.keras.models.load_model('history_lstm.h5')
    print("LSTM model loaded successfully.")
    print("LSTM input shape:", model_lstm.input_shape)
    print("LSTM output shape:", model_lstm.output_shape)
except Exception as e:
    print("Error loading LSTM:", e)

try:
    model_rnn = tf.keras.models.load_model('history_rnn.h5')
    print("RNN model loaded successfully.")
    print("RNN input shape:", model_rnn.input_shape)
    print("RNN output shape:", model_rnn.output_shape)
except Exception as e:
    print("Error loading RNN:", e)
