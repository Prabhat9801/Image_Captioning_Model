import gradio as gr
import tensorflow as tf
import numpy as np
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
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1
print(f"Tokenizer loaded! Vocabulary size: {vocab_size}")

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
print("All models loaded successfully!")

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
    in_text = 'start '
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
    start = [tokenizer.word_index['start']]
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
                prob += np.log(preds[0][w] + 1e-10)  # Use log probabilities
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

# --- Gradio Interface Function ---
def generate_captions_gradio(image):
    """Main function for Gradio interface"""
    if image is None:
        return "Please upload an image.", "Please upload an image."

    try:
        # Ensure image is in PIL format
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract features and generate captions
        image_features = extract_image_features(inception_v3_model, image)
        greedy_cap = greedy_generator(image_features)
        beam_cap = beam_search_generator(image_features, K_beams=3)

        # Format output
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
        - **Beam Search (K=3)**: Higher quality captions with beam search
        
        **Model Architecture**: CNN-RNN (InceptionV3 + LSTM)  
        **Dataset**: Flickr8k
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            generate_btn = gr.Button("Generate Captions", variant="primary", size="lg")
            
        with gr.Column():
            greedy_output = gr.Textbox(label="Greedy Search Result", lines=4)
            beam_output = gr.Textbox(label="Beam Search Result", lines=4)
    
    gr.Markdown(
        """
        ---
        **Created by**: Prabhar Kumar Singh  
        **GitHub**: [Prabhat9801/Image_Captioning_Model](https://github.com/Prabhat9801/Image_Captioning_Model)
        """
    )
    
    # Button click event
    generate_btn.click(
        fn=generate_captions_gradio,
        inputs=[image_input],
        outputs=[greedy_output, beam_output]
    )
    
    # Auto-generate on image upload
    image_input.change(
        fn=generate_captions_gradio,
        inputs=[image_input],
        outputs=[greedy_output, beam_output]
    )

if __name__ == "__main__":
    demo.launch()
