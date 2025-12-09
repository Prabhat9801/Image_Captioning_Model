# 🖼️ Image Captioning Model - Hugging Face Deployment

This folder contains everything needed to deploy the Image Captioning Model on Hugging Face Spaces.

## 📦 Files

- `app.py` - Gradio application
- `requirements.txt` - Python dependencies
- `caption_model_final.keras` - Trained model (66 MB)
- `tokenizer.pkl` - Tokenizer with vocabulary
- `model_config.pkl` - Model configuration (max_caption_length, cnn_output_dim)

## 🚀 Quick Deploy to Hugging Face Spaces

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - Name: `image-captioning-model` (or your choice)
   - SDK: **Gradio**
   - Visibility: Public or Private
4. Upload all files from this folder
5. Wait for build (5-10 minutes)
6. Done! 🎉

## 🔧 Model Architecture

```
Input Image (299x299x3)
    ↓
InceptionV3 (pretrained)
    ↓
Features (2048-dim) → BatchNorm → Dense(256) → BatchNorm
                                                    ↓
Caption Sequence → Embedding(256) → LSTM(256) -----+
                                                    ↓
                                                  Add
                                                    ↓
                                              Dense(256)
                                                    ↓
                                          Dense(vocab_size, softmax)
```

## 📊 Model Details

- **CNN**: InceptionV3 (pretrained on ImageNet)
- **RNN**: LSTM with 256 units
- **Embedding**: 256 dimensions
- **Vocabulary Size**: ~8,586 words
- **Max Caption Length**: 34 words
- **Dataset**: Flickr8k
- **Training**: 15 epochs with early stopping

## 🎯 Caption Generation Methods

1. **Greedy Search**: Fast, selects most probable word at each step
2. **Beam Search (K=3)**: Higher quality, explores top 3 candidates

## 💡 Usage

Upload an image and the model will generate:
- A greedy search caption (fast)
- A beam search caption (higher quality)

## 📝 Technical Notes

- Uses TensorFlow 2.15.0
- Gradio 4.44.0 for the interface
- Model saved in Keras 3.0 format (.keras)
- Tokenizer saved as pickle file

---

**Created by**: Prabhar Kumar Singh  
**GitHub**: [Prabhat9801/Image_Captioning_Model](https://github.com/Prabhat9801/Image_Captioning_Model)
