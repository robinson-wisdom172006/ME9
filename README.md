# Tomato Leaf Disease Classifier — GET 324 Mini-Project (Group ME9)

**Task:** Binary image classification — *Tomato Leaf Mold* vs *Tomato Septoria Leaf Spot*
**Course:** GET 324 — Cloud Computing and AI Model Deployment for Engineering Applications
**Department:** Mechanical Engineering

## 1. Overview
This project trains a Convolutional Neural Network (using MobileNetV2 transfer
learning) to distinguish between two tomato leaf diseases from a photo, and
deploys the model as a Streamlit web application.

## 2. Project Structure
```
.
├── app.py                 # Streamlit web application (deployment entry point)
├── train_model.py         # Script to train and save the CNN model
├── requirements.txt       # Python dependencies
├── tomato_model.h5        # Trained model (generated after running train_model.py)
├── class_names.txt        # Maps model output index to class name (auto-generated)
├── training_history.png   # Accuracy/loss curves (auto-generated, for report)
├── dataset/                # (not included in repo — see Dataset section)
└── README.md
```

## 3. Dataset
Source: **PlantVillage Dataset** (Kaggle) — https://www.kaggle.com/datasets/emmarex/plantdisease

We used the two classes:
- `Tomato___Leaf_Mold`
- `Tomato___Septoria_leaf_spot`

Folder layout expected by `train_model.py`:
```
dataset/
├── Tomato_Leaf_Mold/
└── Tomato_Septoria_Leaf_Spot/
```
> The `dataset/` folder is **not pushed to GitHub** (too large) — it's listed in `.gitignore`.
> Anyone reproducing this project should download it from Kaggle directly.

## 4. How to Run Locally
```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Only if retraining) Prepare dataset/ folder as described above, then:
python train_model.py

# 5. Run the app
streamlit run app.py
```
The app opens at `http://localhost:8501`.

## 5. How to Use the App
1. Open the deployed app link (or run locally).
2. Click "Choose a leaf image..." and upload a `.jpg`/`.jpeg`/`.png` photo of a tomato leaf.
3. The app displays the predicted class (Leaf Mold or Septoria Leaf Spot) and a confidence score.

## 6. Deployment (Streamlit Community Cloud)
1. Push this project (including `tomato_model.h5` and `class_names.txt`) to a public GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** → select your repo/branch → set **Main file path** to `app.py`.
4. Click **Deploy**.
5. If `tomato_model.h5` is larger than GitHub's 100MB limit, use **Git LFS**:
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes tomato_model.h5
   git commit -m "Track model with LFS"
   git push
   ```

## 7. Model Details
- **Base model:** MobileNetV2 (pre-trained on ImageNet)
- **Approach:** Transfer learning — frozen base + custom classification head,
  followed by fine-tuning of the last 30 layers of the base model
- **Input size:** 224 × 224 × 3
- **Output:** Single sigmoid unit (binary classification)
- **Loss:** Binary cross-entropy
- **Optimizer:** Adam

## 8. Report
See [report.md](./report.md) for the 100–150 word summary covering dataset source, usage,
challenges, and improvements.

## 9. Team Members
Check [CONTRIBUTORS.md](./CONTRIBUTORS.md) to see the list of people in GROUP ME9 that contributed to this project
