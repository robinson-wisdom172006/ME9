# Project Report — Group ME9

**Task:** Tomato Leaf Mold vs Tomato Septoria Leaf Spot Classification

> Fill in the bracketed parts after training and deployment. Keep the final
> version between 100–150 words.

The dataset used for this project was sourced from the PlantVillage dataset
on Kaggle, containing labeled images of tomato leaves affected by Leaf Mold
and Septoria Leaf Spot. We built a binary image classifier using transfer
learning with a pre-trained MobileNetV2 model, fine-tuned on [NUMBER] images
per class, achieving a validation accuracy of [XX.X]%. The trained model was
deployed as an interactive Streamlit web application, allowing users to
upload a leaf image and instantly receive a predicted diagnosis with a
confidence score. During development, we encountered [DESCRIBE CHALLENGE,
e.g. "class imbalance in the raw dataset" or "GitHub's 100MB file size limit
when pushing the trained model"], which we resolved by [DESCRIBE SOLUTION,
e.g. "applying data augmentation" or "using Git LFS"]. Future improvements
could include [e.g. "expanding to more disease classes" or "adding
Grad-CAM visualizations to explain predictions"].

**Word count: [XXX]**
