# 🧪 Local Testing Guide

Follow these steps to test the deployment locally before uploading to Hugging Face Spaces.

---

## 📋 Prerequisites

- Python 3.10 installed
- At least 4GB RAM available
- Internet connection (for downloading InceptionV3 weights)

---

## 🚀 Step-by-Step Testing

### Step 1: Navigate to Deployment Folder

```powershell
cd c:\Users\prabh\Desktop\Image_Captioning_Model\huggingface_deployment
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```powershell
python -m venv venv_test
```

### Step 3: Activate Virtual Environment

```powershell
.\venv_test\Scripts\Activate
```

You should see `(venv_test)` in your terminal prompt.

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- tensorflow==2.15.0 (~500 MB)
- gradio==4.44.0
- pillow==10.3.0
- numpy==1.26.4

**Note**: This may take 5-10 minutes depending on your internet speed.

### Step 5: Run the App

```powershell
python app.py
```

### Step 6: Test in Browser

The app will start and show:
```
Running on local URL:  http://127.0.0.1:7860
```

1. Open your browser and go to: **http://127.0.0.1:7860**
2. Upload a test image
3. Wait for captions to generate
4. Verify both Greedy and Beam Search captions appear

### Step 7: Stop the App

Press `Ctrl+C` in the terminal to stop the app.

### Step 8: Deactivate Virtual Environment

```powershell
deactivate
```

---

## ✅ What to Check

### Model Loading
- [ ] "Loading model configuration..." appears
- [ ] "Loading tokenizer..." appears
- [ ] "Tokenizer loaded! Vocabulary size: 8586" appears
- [ ] "Loading InceptionV3 model..." appears
- [ ] "Loading caption generation model..." appears
- [ ] "All models loaded successfully!" appears

### Image Upload
- [ ] Can upload image via drag-and-drop
- [ ] Can upload image via file browser
- [ ] Image preview shows correctly

### Caption Generation
- [ ] Greedy Search caption generates within 5-10 seconds
- [ ] Beam Search caption generates within 10-15 seconds
- [ ] Captions are relevant to the image
- [ ] No error messages appear

### UI/UX
- [ ] Interface looks clean and professional
- [ ] Buttons are clickable
- [ ] Text is readable
- [ ] No layout issues

---

## 🐛 Common Issues & Solutions

### Issue 1: "No module named 'tensorflow'"
**Solution**: Make sure you activated the virtual environment and ran `pip install -r requirements.txt`

### Issue 2: Model loading takes too long
**Solution**: This is normal for the first run. TensorFlow downloads InceptionV3 weights (~92 MB) on first use.

### Issue 3: "Out of memory" error
**Solution**: Close other applications. The model needs ~2-3 GB RAM.

### Issue 4: Gradio doesn't open in browser
**Solution**: Manually open http://127.0.0.1:7860 in your browser.

### Issue 5: Captions are gibberish
**Solution**: This might indicate model/tokenizer mismatch. Verify all files are from the same training run.

---

## 📊 Expected Performance

- **Model Loading Time**: 30-60 seconds (first time)
- **Image Upload**: Instant
- **Feature Extraction**: 1-2 seconds
- **Greedy Caption**: 3-5 seconds
- **Beam Search Caption**: 8-12 seconds
- **Total Time per Image**: 10-15 seconds

---

## 🎯 Test Images

Try these types of images:
1. **Simple scenes**: Single object (e.g., a dog, car)
2. **Complex scenes**: Multiple objects (e.g., people in a park)
3. **Indoor scenes**: Room, kitchen, office
4. **Outdoor scenes**: Beach, mountain, street
5. **Action scenes**: Person running, playing sports

---

## ✨ Success Criteria

Your deployment is ready for Hugging Face if:
- ✅ App starts without errors
- ✅ Models load successfully
- ✅ Captions generate for test images
- ✅ Captions are meaningful (not random words)
- ✅ UI is responsive and looks good
- ✅ No crashes or freezes

---

## 📝 Next Steps After Successful Testing

1. Stop the local app (Ctrl+C)
2. Deactivate virtual environment
3. Upload files to Hugging Face Spaces:
   - app.py
   - requirements.txt
   - caption_model_final.keras
   - tokenizer.pkl
   - model_config.pkl
   - README.md (optional)

---

## 💡 Tips

- **First Run**: Will be slower due to downloading InceptionV3 weights
- **Subsequent Runs**: Much faster as weights are cached
- **RAM Usage**: Monitor task manager - should use 2-3 GB
- **GPU**: Not required, runs fine on CPU
- **Internet**: Only needed for first run (InceptionV3 download)

---

**Happy Testing!** 🎉

If everything works locally, it will work on Hugging Face Spaces!
