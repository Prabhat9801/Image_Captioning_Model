import gradio as gr
import tensorflow as tf
import numpy as np
import json
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image

# --- Configuration paths ---
MODEL_WEIGHTS_PATH = "model_weights.h5"
TOKENIZER_DATA_PATH = "tokenizer_data.json"
CONFIG_PATH = "model_config.pkl"

# --- Load configurations ---
print("Loading model configuration...")
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)
max_caption_length = config['max_caption_length']
cnn_output_dim = config['cnn_output_dim']

# --- Load tokenizer data from JSON ---
print("Loading tokenizer data...")
with open(TOKENIZER_DATA_PATH, 'r') as f:
    tokenizer_data = json.load(f)

# Create a simple tokenizer class
class SimpleTokenizer:
    def __init__(self, word_index, index_word):
        self.word_index = word_index
        self.index_word = {int(k): v for k, v in index_word.items()}
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_index.get(word, 0) for word in words]
            sequences.append(sequence)
        return sequences

# Initialize tokenizer
tokenizer = SimpleTokenizer(tokenizer_data['word_index'], tokenizer_data['index_word'])
vocab_size = len(tokenizer.word_index) + 1
print(f"Tokenizer loaded! Vocabulary size: {vocab_size}")

# --- Load InceptionV3 for feature extraction ---
print("Loading InceptionV3 model...")
inception_v3_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
inception_v3_model = tf.keras.Model(
    inputs=inception_v3_model.inputs, 
    outputs=inception_v3_model.layers[-2].output
)

# --- Rebuild the caption model architecture ---
print("Building caption model architecture...")

# Image feature input
image_features_input = Input(shape=(cnn_output_dim,), name='Features_Input')
image_features_bn = BatchNormalization()(image_features_input)
image_features_dense = Dense(256, activation='relu')(image_features_bn)
image_features_bn2 = BatchNormalization()(image_features_dense)

# Sequence input
sequence_input = Input(shape=(max_caption_length,), name='Sequence_Input')
sequence_embedding = Embedding(vocab_size, 256, mask_zero=True)(sequence_input)
sequence_lstm = LSTM(256)(sequence_embedding)

# Merge features
merged = Add()([image_features_bn2, sequence_lstm])
merged_dense = Dense(256, activation='relu')(merged)
output = Dense(vocab_size, activation='softmax', name='Output_Layer')(merged_dense)

# Create model
caption_model = Model(inputs=[image_features_input, sequence_input], outputs=output)

# Load weights
print("Loading model weights...")
caption_model.load_weights(MODEL_WEIGHTS_PATH)
print("Model loaded successfully!")

# --- Utility functions ---
def preprocess_image(image_pil):
    """Preprocess PIL image for InceptionV3"""
    img = image_pil.resize((299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def extract_image_features(model, image_pil):
    """Extract features from image using InceptionV3"""
    img_processed = preprocess_image(image_pil)
    features = model.predict(img_processed, verbose=0)
    return features.flatten()

def greedy_generator(image_features):
    """Generate caption using greedy search"""
    in_text = 'start'
    for _ in range(max_caption_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length).reshape((1, max_caption_length))
        prediction = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)
        idx = np.argmax(prediction)
        
        if idx == 0 or idx not in tokenizer.index_word:
            break
            
        word = tokenizer.index_word[idx]
        in_text += ' ' + word
        
        if word == 'end':
            break

    in_text = in_text.replace('start ', '').replace(' start', '')
    in_text = in_text.replace(' end', '').replace('end', '')
    return in_text.strip()

def beam_search_generator(image_features, K_beams=3):
    """Generate caption using beam search"""
    start = [tokenizer.word_index.get('start', 1)]
    start_word = [[start, 0.0]]
    
    for _ in range(max_caption_length):
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_caption_length).reshape((1, max_caption_length))
            preds = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)
            word_preds = np.argsort(preds[0])[-K_beams:]
            
            for w in word_preds:
                if w == 0:
                    continue
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += np.log(preds[0][w] + 1e-10)
                temp.append([next_cap, prob])

        start_word = temp
        start_word = sorted(start_word, reverse=True, key=lambda l: l[1])
        start_word = start_word[:K_beams]

    best_caption_sequence = start_word[0][0]
    captions_ = []
    
    for i in best_caption_sequence:
        if i in tokenizer.index_word:
            captions_.append(tokenizer.index_word[i])
    
    final_caption = []
    for i in captions_:
        if i != 'end' and i != 'start':
            final_caption.append(i)
        elif i == 'end':
            break

    return ' '.join(final_caption).strip()

# --- Gradio Interface ---
def generate_captions_gradio(image):
    """Main function for Gradio interface"""
    if image is None:
        return "Please upload an image.", "Please upload an image."

    try:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_features = extract_image_features(inception_v3_model, image)
        greedy_cap = greedy_generator(image_features)
        beam_cap = beam_search_generator(image_features, K_beams=3)

        greedy_output = f"**Greedy Search:**\n\n{greedy_cap.capitalize()}"
        beam_output = f"**Beam Search (K=3):**\n\n{beam_cap.capitalize()}"

        return greedy_output, beam_output
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        return error_msg, error_msg

# --- Gradio App ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🖼️ Image Captioning Model
        ### Generate natural language descriptions for your images
        
        Upload an image and get captions using two different algorithms:
        - **Greedy Search**: Fast caption generation
        - **Beam Search (K=3)**: Higher quality captions
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            generate_btn = gr.Button("Generate Captions", variant="primary")
            
        with gr.Column():
            greedy_output = gr.Textbox(label="Greedy Search Result", lines=3)
            beam_output = gr.Textbox(label="Beam Search Result", lines=3)
    
    gr.Markdown(
        """
        ---
        **Model**: CNN-RNN (InceptionV3 + LSTM) | **Dataset**: Flickr8k
        
        **Created by**: Prabhar Kumar Singh | [GitHub](https://github.com/Prabhat9801/Image_Captioning_Model)
        """
    )
    
    generate_btn.click(
        fn=generate_captions_gradio,
        inputs=[image_input],
        outputs=[greedy_output, beam_output]
    )
    
    image_input.change(
        fn=generate_captions_gradio,
        inputs=[image_input],
        outputs=[greedy_output, beam_output]
    )

if __name__ == "__main__":
    demo.launch()
