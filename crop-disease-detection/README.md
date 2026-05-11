# Crop Disease Detection and Classification for Farmers

This project mirrors the structure of `helmet-detection` and is adapted for:
- Crop disease classification
- K-Means clustering + Otsu segmentation
- PlantVillage-style dataset folders

## Dataset Location

Place PlantVillage images under:

Data/crop_disease/raw/

Expected structure:

Data/crop_disease/raw/
  Apple___Apple_scab/
    image1.jpg
  Apple___Black_rot/
    image2.jpg
  ...

## Project Structure

crop-disease-detection/
  dataset/
    labels/
  preprocessing/
    resize.py
    denoise.py
    enhance.py
  segmentation/
    kmeans_otsu.py
  feature_extraction/
    color_texture_features.py
  classification/
    train_classifier.py
    predict_image.py
  evaluation/
    metrics.py
  runs/
    classify/

## Quick Start

1) Install dependencies

pip install -r crop-disease-detection/requirements.txt

2) Resize raw dataset

python crop-disease-detection/preprocessing/resize.py

3) Apply denoise and enhancement (optional)

python crop-disease-detection/preprocessing/denoise.py
python crop-disease-detection/preprocessing/enhance.py

4) Run K-Means + Otsu segmentation

python crop-disease-detection/segmentation/kmeans_otsu.py

5) Extract features

python crop-disease-detection/feature_extraction/color_texture_features.py

6) Train classifier

python crop-disease-detection/classification/train_classifier.py

7) Predict one image

python crop-disease-detection/classification/predict_image.py --image "path/to/leaf.jpg"

## Notes

- By default, scripts read and write inside Data/crop_disease.
- You can override paths using command line arguments in each script.
