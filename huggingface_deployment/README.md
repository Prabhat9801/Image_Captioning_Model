# Image Captioning Model - Hugging Face Deployment

This folder contains everything needed to deploy the Image Captioning Model on Hugging Face Spaces.

## 📦 Files

- `app.py` - Gradio application (rebuilds model architecture and loads weights)
- `requirements.txt` - Python dependencies
- `model_weights.h5` - Model weights only (22 MB)
- `tokenizer_data.json` - Tokenizer vocabulary (JSON format)
- `model_config.pkl` - Model configuration

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

## 💡 How It Works

This deployment uses a **different approach** than typical model loading:

- **Instead of**: Loading the full H5 model (which has compatibility issues)
- **We do**: Rebuild the model architecture in code and load only the weights

This avoids all the Keras version compatibility issues!

## 🔧 Technical Details

- **TensorFlow**: 2.10.0
- **Gradio**: 3.50.2
- **Model Architecture**: CNN-RNN (InceptionV3 + LSTM)
- **Weights File**: 22 MB (much smaller than full model)
- **Tokenizer**: JSON format (no pickle compatibility issues)

## ✅ Advantages of This Approach

1. ✅ No Keras version conflicts
2. ✅ Smaller file size (22 MB vs 66 MB)
3. ✅ Works with any TensorFlow 2.x version
4. ✅ Easy to modify model architecture if needed
5. ✅ No pickle security concerns

## 📝 Notes

- The model architecture is defined in `app.py`
- Weights are loaded using `model.load_weights()`
- This is the recommended approach for deploying Keras models

---

**Created by**: Prabhar Kumar Singh  
**GitHub**: [Prabhat9801/Image_Captioning_Model](https://github.com/Prabhat9801/Image_Captioning_Model)
