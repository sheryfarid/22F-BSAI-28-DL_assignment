# 22F-BSAI-28-DL_assignment

## View Notebook
👉 [Click here to view notebook](https://colab.research.google.com/drive/1rcs-gjkgWtfegeSnm2j9Fptj1GwbxWfE?usp=sharing)

# 🏥 Clinical Early Warning System
### Deep Learning Pipeline for Sepsis Prediction

**22F-BSAI-28 | Deep Learning Assignment**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rcs-gjkgWtfegeSnm2j9Fptj1GwbxWfE?usp=sharing)


---

## 📌 Project Overview

Sepsis kills **1 in 5 ICU patients**. Every hour of delayed treatment raises mortality by **7%**. This project builds a three-generation deep learning pipeline that predicts sepsis onset **6 hours early** from ICU patient data.

The pipeline progresses through three generations of increasing complexity, each addressing the limitations of the previous:

| Generation | Model | Core Idea |
|---|---|---|
| Gen 1 | DNN (SGD vs Adam) | Flat tabular vital signs |
| Gen 2 | LSTM / Bi-LSTM / GRU | Temporal trends over 24 hours |
| Gen 3 | ClinicalBERT | Clinical language understanding |


---

## 📊 Dataset

**PhysioNet Sepsis Prediction Challenge 2019**

| Property | Value |
|---|---|
| Total rows | 1,552,210 hourly ICU readings |
| Features | 40 (vitals, labs, demographics) |
| Sepsis cases | ~1.8% of all rows |
| Patients | ~40,000+ unique ICU stays |

The severe **98.2% / 1.8% class imbalance** is the central challenge. A model predicting "no sepsis" for every patient scores 98% accuracy yet catches **zero** real cases — this is the accuracy trap. For this reason, **Recall is the primary metric** throughout the project.

| Metric | What It Measures | Clinical Importance |
|---|---|---|
| **Recall** | Sepsis cases correctly identified | **MOST CRITICAL** — missed = patient at risk |
| Precision | Correct among all raised alarms | Secondary — false alarms cost resources |
| Accuracy | Overall correct predictions | **MISLEADING** with 98/2 imbalance |
| F1 Score | Balance of precision and recall | Useful overall summary |

---

## 🏗️ Pipeline Architecture

```
Dataset.csv (PhysioNet 2019)
        │
        ▼
┌─────────────────────────────────────────┐
│         PART 1 — Feature Selection      │
│  • Drop >70% missing columns (27 dropped)│
│  • Random Forest importance scoring     │
│  • Point-Biserial correlation           │
│  • Final 11 features selected           │
└──────────────────┬──────────────────────┘
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │  Gen 1  │ │  Gen 2  │ │  Gen 3  │
  │   DNN   │ │  LSTM/  │ │Clinical │
  │SGD+Adam │ │BiLSTM/  │ │  BERT   │
  │         │ │  GRU    │ │         │
  └────┬────┘ └────┬────┘ └────┬────┘
       └───────────┴───────────┘
                   │
                   ▼
        Final Comparison Table
         + Deployment Decision
```

---

## ⚙️ Generation 1 — Deep Neural Network

**Architecture:** `Dense(128) → BN → Dropout(0.3) → Dense(64) → BN → Dropout(0.3) → Dense(32) → BN → Dropout(0.2) → Sigmoid`

- **BatchNormalization**: normalises layer activations, stabilises gradients, allows higher learning rates
- **Dropout**: randomly disables neurons during training to prevent overfitting
- **Class weight 27.8×**: every missed sepsis case penalised 27.8× harder, directly training for Recall
- **Compared**: SGD (lr=0.01, momentum=0.9) vs Adam (lr=0.001) over 50 epochs with early stopping

### Gen 1 Results

| Model | Accuracy | Precision | Recall | F1 | Train Time |
|---|---|---|---|---|---|
| DNN + SGD | 0.8169 | 0.0553 | 0.5712 | 0.1009 | 59s |
| **DNN + Adam** | 0.7816 | 0.0509 | **0.6311** | 0.0942 | 315s |

**Key finding:** Adam catches 251 more sepsis patients than SGD (Recall 0.631 vs 0.571). Missing 37% because DNN treats each hourly reading independently — no concept of trends.

---

## ⏱️ Generation 2 — Time Series Models

Sepsis develops gradually — a rising heart rate trend over 12 hours predicts sepsis far better than any single reading. Gen 2 fixes the DNN's core limitation by processing **patient sequences**.

**Data preparation:** All rows grouped by `Patient_ID`, sorted by `Hour`. Last **24 hours** form one sequence (shape: `24 timesteps × 11 features`). Patients with fewer than 24 hours are zero-padded. At patient level, class balance improves to 92.7% vs 7.3% (class weight 6.85×).

### Gen 2 Results

| Model | Accuracy | Precision | Recall | F1 | Train Time |
|---|---|---|---|---|---|
| LSTM | 0.8973 | 0.3820 | 0.6616 | 0.4844 | 86s |
| **Bidirectional LSTM** | 0.8858 | 0.3595 | **0.7256** | 0.4808 | 186s |
| GRU | 0.8989 | 0.3896 | 0.6829 | 0.4961 | **22s** |

**Key finding:** Recall jumps from 0.631 → 0.726 (+0.095). Temporal trend modelling is the key driver. Bi-LSTM crosses the 0.70 clinical target but cannot run in real-time — it needs future data. **GRU recommended for live deployment**: Recall 0.683, trains in 22s, no GPU needed.

---

## 🧠 Generation 3 — ClinicalBERT

**Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) pre-trained on **MIMIC-III** real hospital notes already understands tachycardia, hypotension, and clinical sepsis indicators.

Since PhysioNet contains only vital signs (no free text), **synthetic clinical notes** are generated from vitals using Sepsis-3 clinical thresholds. The word "sepsis" never appears in notes to prevent data leakage.

```
HR=105  →  "Tachycardia noted, heart rate 105 bpm."
SBP=88  →  "Hypotension detected, systolic BP 88 mmHg."
Temp=38.5 → "Fever present, temperature 38.5°C, infection suspected."
```

### Fine-tuning Strategies

| Strategy | Trainable Params | Recall | Accuracy | Train Time |
|---|---|---|---|---|
| Frozen (head only) | 1,538 / 108M | 0.250 | 0.8600 | 1,326s |
| **Full Fine-Tune** | **108M / 108M** | **0.500** | **0.9507** | 6,239s |

**Key finding:** Frozen BERT (Recall 0.250) is unsuitable — the fixed representations cannot adapt. Full fine-tune achieves best Precision (0.649) across all 7 models. Lower Recall than Gen 2 because only 5,000 synthetic notes were used — BERT needs real clinical text at scale to reach its potential.

### Attention Heatmap Interpretation

The attention visualisation confirms the model focuses on clinically meaningful tokens (`tachycardia`, `temperature`, `hypotension`) and ignores filler words (`at`, `per`, `of`) — important for clinician trust.

---

## 📈 Final Results — All 7 Models

| Model | Accuracy | Precision | Recall | F1 | Time | Gen |
|---|---|---|---|---|---|---|
| DNN + SGD | 0.817 | 0.055 | 0.571 | 0.101 | 59s | 1 |
| DNN + Adam | 0.782 | 0.051 | 0.631 | 0.094 | 315s | 1 |
| LSTM | 0.897 | 0.382 | 0.662 | 0.484 | 86s | 2 |
| **Bidirectional LSTM** | 0.886 | 0.360 | **0.726** | 0.481 | 186s | 2 |
| **GRU** ⭐ | 0.899 | 0.390 | 0.683 | 0.496 | **22s** | 2 |
| ClinicalBERT Frozen | 0.860 | 0.148 | 0.250 | 0.186 | 1,326s | 3 |
| ClinicalBERT Full FT | **0.951** | **0.649** | 0.500 | 0.565 | 6,239s | 3 |

> Only **Bi-LSTM (0.726)** crosses the 0.70 Recall clinical target. **GRU** is the best real-time model (0.683 in 22s).

---

## 🚀 Deployment Decision

**Recommended pipeline for live ICU deployment:**

```
Real-time stream of hourly vitals
        │
        ▼
   ┌─────────┐     Recall 0.683
   │   GRU   │ ──► Hourly alert every hour
   │ (22s)   │     No GPU required
   └────┬────┘
        │  Alert triggered
        ▼
┌──────────────┐   Precision 0.649
│ ClinicalBERT │──► Confirmation from
│  Full FT     │    synthetic note analysis
└──────────────┘
```

- **GRU** for continuous real-time monitoring — Recall 0.683, trains in 22s, generates hourly alerts
- **Bi-LSTM** for offline retrospective analysis only — needs future data, not deployable live
- **ClinicalBERT Full FT** as secondary confirmation alongside GRU

---

## ⚠️ Ethics & Limitations

- GRU still **misses 31.7%** of sepsis patients — must be a support tool, not a replacement
- **Clinicians must be able to override** the system at any time
- **Fairness audits** across age, gender, and demographics required before deployment
- Synthetic notes are a limitation — real clinical text would significantly improve Gen 3 Recall
- Class imbalance means Accuracy is a misleading metric throughout

---

## 🔬 Multimodal Extension (Future Work)

Adding chest X-rays creates a full multimodal system:

```
Vitals ──► GRU Branch ──────────────────┐
X-Rays ──► DenseNet-121 Branch ─────────┼──► Fusion Layer ──► Final Classifier
Notes  ──► ClinicalBERT Branch ─────────┘
```

Multimodal models consistently outperform single modalities. Key challenges: timestamp alignment and GPU compute requirements.

---

## 🛠️ Setup & Usage

### Requirements

```bash
pip install tensorflow torch transformers scikit-learn pandas numpy \
            matplotlib seaborn imbalanced-learn tqdm
```

### Run the Notebook

```bash
# Option 1: Google Colab (recommended)
# Open: https://colab.research.google.com/drive/1rcs-gjkgWtfegeSnm2j9Fptj1GwbxWfE?usp=sharing

# Option 2: Local
git clone https://github.com/22F-BSAI-28/22F-BSAI-28-DL_assignment.git
cd 22F-BSAI-28-DL_assignment
jupyter notebook main_.ipynb
```

### Data

Place `Dataset.csv` (PhysioNet Sepsis Challenge 2019, combined training sets A+B) in the same directory as the notebook. The preprocessing pipeline will auto-generate `Dataset_final.csv`.



## 👤 Author

**22F-BSAI-28**
BS Artificial Intelligence
Deep Learning Assignment — Three-Generation Clinical AI Pipeline
