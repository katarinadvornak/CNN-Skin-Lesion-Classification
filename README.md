# CNN Skin Lesion Classification

This project provides a Convolutional Neural Network (CNN) for classifying skin lesion images. It supports training and testing with different strategies, including class weighting, patient features, and data augmentation. A VGG-16-based model is also included for transfer learning experiments.

---

## Dataset

Download the dataset from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).  
For quick testing, we recommend resizing images to **28x28**.

---

## Available Models

| Model | Description | Image Size |
|-------|------------|------------|
| `skin_disease_model.h5` | Prototype, no features, unbalanced dataset | 64x64 |
| `skin_disease_model64x64.features_weights.keras` | Trained on patient features and images with class weights | 64x64 |
| `skin_disease_model64x64.featuresaugmented.keras` | Trained on patient features and augmented images with class weights | 64x64 |
| `TEST64x64aug.keras` | Trained on augmented images only, no patient features | 64x64 |

---

## Training

### 1. Class Weights Only
1. Edit `training_with_weights.py` to set image paths, parameters, and model path.
2. Configure image size in `loadimages.py`.
3. Run the training script.

### 2. Class Weights + Patient Features
1. Edit `training_with_features_weights.py` to set image paths and model path.
2. Configure image size in `loadimages_features_weights.py`.
3. Run the training script.

### 3. Augmentation Only
1. Use `AugmentedImageGenerator` to generate augmented images.
2. Edit `training_with_augmentation.py` with parameters and model path.
3. Configure image size in `loadimages.py`.
4. Run the training script.

### 4. Augmentation + Patient Features
1. Generate augmented images with `AugmentedImageGenerator`.
2. Edit `training_with_features_augmentation.py` to set image paths, CSV patient data paths, and model path.
3. Configure image size in `loadimages_features_aug.py`.
4. Run the training script.

---

## Testing

### On Unseen Data
1. Edit paths and image size in `cnn_tester.py`.
2. Run the script to get results.

### Single Image Prediction
1. Edit paths, model, and image ID in `Main.py`.
2. Adjust image size in `predictinglogic.py`.
3. Run `Main.py` to predict a specific image.

---

## VGG-16 Models

### Initial Training (Frozen Layers)
1. Edit `vgg16_all_frozen.py` for image paths and parameters.
2. Image size is **128x128**.
3. Run the script.

### Fine-Tuning (Unfrozen Layers)
1. Edit `vgg16_finetuning.py` for image paths and parameters.
2. Image size is **128x128**.
3. Run the script.

### Testing
1. Edit `vgg16_evaluation.py` for test images and model.
2. Image size is **128x128**.
3. Run the script.

---

## Notes
- Models are automatically created if the specified path does not exist.  
- Recommended starting point for quick testing: **28x28 images with class weights**.  

---

## License
[MIT License](LICENSE)
