import gradio as gr
import tensorflow as tf
import numpy as np
import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from PIL import Image

# --- Configuration paths ---
MODEL_PATH = "caption_model_final.keras"
TOKENIZER_PATH = "tokenizer.pkl"
CONFIG_PATH = "model_config.pkl"

# --- Load configurations ---
print("Loading model configuration...")
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)
max_caption_length = config['max_caption_length']
cnn_output_dim = config['cnn_output_dim']

# --- Load tokenizer ---
print("Loading tokenizer...")
import sys

# Add compatibility for legacy Keras tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
except ModuleNotFoundError as e:
    if 'keras.src.legacy' in str(e) or 'keras' in str(e):
        # Handle legacy Keras compatibility issue
        print("Handling legacy Keras compatibility...")
        import keras
        sys.modules['keras.preprocessing'] = keras.preprocessing
        sys.modules['keras.preprocessing.text'] = keras.preprocessing.text
        
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        raise

vocab_size = len(tokenizer.word_index) + 1

# --- Load the pre-trained InceptionV3 model for feature extraction ---
print("Loading InceptionV3 model...")
inception_v3_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
inception_v3_model = tf.keras.Model(
    inputs=inception_v3_model.inputs, 
    outputs=inception_v3_model.layers[-2].output
)

# --- Load the captioning model ---
print("Loading caption generation model...")
caption_model = load_model(MODEL_PATH, compile=False)
print("✅ All models loaded successfully!")


# --- Define utility functions ---
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
    """Generate caption using greedy search algorithm"""
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

    # Clean up the caption
    in_text = in_text.replace('start ', '').replace(' start', '')
    in_text = in_text.replace(' end', '').replace('end', '')
    return in_text.strip()


def beam_search_generator(image_features, K_beams=3):
    """Generate caption using beam search algorithm"""
    start = [tokenizer.word_index.get('start', 1)]
    start_word = [[start, 0.0]]
    
    for _ in range(max_caption_length):
        temp = []
        for s in start_word:
            sequence = pad_sequences([s[0]], maxlen=max_caption_length).reshape((1, max_caption_length))
            preds = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)
            word_preds = np.argsort(preds[0])[-K_beams:]
            
            for w in word_preds:
                if w == 0:  # Skip padding token
                    continue
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += np.log(preds[0][w] + 1e-10)  # Add small epsilon to avoid log(0)
                temp.append([next_cap, prob])

        start_word = temp
        # Sort by probability (higher is better for log-probs) and keep top K_beams
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

    final_caption = ' '.join(final_caption)
    return final_caption.strip()


# --- Gradio Interface Function ---
def generate_captions_gradio(image):
    """Main function for Gradio interface"""
    if image is None:
        return "⚠️ Please upload an image.", "⚠️ Please upload an image."

    try:
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if image is in different mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract features
        image_features = extract_image_features(inception_v3_model, image)

        # Generate captions
        greedy_cap = greedy_generator(image_features)
        beam_cap = beam_search_generator(image_features, K_beams=3)

        # Format output
        greedy_output = f"🔍 **Greedy Search Caption:**\n\n{greedy_cap.capitalize()}"
        beam_output = f"🎯 **Beam Search Caption (K=3):**\n\n{beam_cap.capitalize()}"

        return greedy_output, beam_output
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg


# --- Custom CSS for better styling ---
custom_css = """
#component-0 {
    max-width: 900px;
    margin: auto;
    padding: 20px;
}
.gradio-container {
    font-family: 'Inter', sans-serif;
}
h1 {
    text-align: center;
    color: #2563eb;
    margin-bottom: 10px;
}
.description {
    text-align: center;
    color: #64748b;
    margin-bottom: 20px;
}
"""

# --- Gradio App ---
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🖼️ Image Captioning Model
        ### Generate natural language descriptions for your images using deep learning
        
        This model uses **InceptionV3** for feature extraction and **LSTM** for caption generation.
        Upload an image and get captions using two different algorithms:
        - **Greedy Search**: Fast, deterministic caption generation
        - **Beam Search (K=3)**: Higher quality captions with beam width of 3
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", 
                label="📤 Upload Image",
                height=400
            )
            generate_btn = gr.Button(
                "✨ Generate Captions", 
                variant="primary",
                size="lg"
            )
            
        with gr.Column(scale=1):
            greedy_output = gr.Textbox(
                label="Greedy Search Result",
                lines=4,
                interactive=False
            )
            beam_output = gr.Textbox(
                label="Beam Search Result",
                lines=4,
                interactive=False
            )
    
    gr.Markdown(
        """
        ---
        ### 📊 About the Model
        
        - **Architecture**: CNN-RNN (InceptionV3 + LSTM)
        - **Training Dataset**: Flickr8k (8,000 images with 40,000 captions)
        - **Vocabulary Size**: ~8,000 words
        - **Max Caption Length**: Variable (typically 20-30 words)
        
        ### 🎯 How it Works
        
        1. **Feature Extraction**: InceptionV3 extracts visual features from the image
        2. **Caption Generation**: LSTM network generates word-by-word descriptions
        3. **Decoding**: Two algorithms produce different quality captions
        
        ### 💡 Tips for Best Results
        
        - Use clear, well-lit images
        - Images with common objects work best
        - The model performs well on everyday scenes and activities
        
        ---
        
        **Created by**: Prabhar Kumar Singh  
        **GitHub**: [Prabhat9801/Image_Captioning_Model](https://github.com/Prabhat9801/Image_Captioning_Model)
        """
    )
    
    # Event handlers
    generate_btn.click(
        fn=generate_captions_gradio,
        inputs=[image_input],
        outputs=[greedy_output, beam_output]
    )
    
    # Also trigger on image upload
    image_input.change(
        fn=generate_captions_gradio,
        inputs=[image_input],
        outputs=[greedy_output, beam_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
