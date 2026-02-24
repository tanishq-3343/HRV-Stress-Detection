#  Heart Rate Variability (HRV) Analysis for Stress & Sleep Monitoring

##  Overview

This project performs **Heart Rate Variability (HRV) analysis** to monitor and classify stress and sleep conditions using RR interval data.

Heart Rate Variability reflects the activity of the **Autonomic Nervous System (ANS)** and is widely used in stress detection, sleep research, wearable devices, and healthcare analytics.

The project includes preprocessing, HRV feature extraction (time & frequency domain), machine learning modeling, and evaluation using standard performance metrics.

---

##  Objectives

- Extract meaningful HRV features from RR intervals  
- Analyze autonomic nervous system behavior  
- Classify stress/sleep states using machine learning  
- Evaluate model performance with multiple metrics  
- Visualize HRV signals and prediction results  

---

##  HRV Features Extracted

###  Time-Domain Features
- Mean RR Interval  
- SDNN (Standard Deviation of NN intervals)  
- RMSSD (Root Mean Square of Successive Differences)  
- pNN50  

###  Frequency-Domain Features
- LF (Low Frequency Power)  
- HF (High Frequency Power)  
- LF/HF Ratio  

These features provide insight into sympathetic and parasympathetic nervous system balance.

---

##  Tech Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Scikit-learn  
- Google Colab  

---

##  Project Structure

HRV-Analysis/
│
├── notebooks/
│   └── HRV_Complete_Project.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   └── model.py
│
├── data/
│   └── (Dataset files or dataset link)
│
├── models/
│   └── saved_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

---

##  How to Run

### ▶ Option 1: Google Colab
1. Upload the notebook  
2. Install required libraries  
3. Run all cells sequentially  

### ▶ Option 2: Run Locally

Install dependencies:

pip install -r requirements.txt

Then run the notebook or training script.

---

##  Model Evaluation Metrics

The model performance is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  

(Add your final metric values here if available.)

---

##  Applications

- Stress monitoring systems  
- Sleep quality analysis  
- Wearable health tracking  
- Clinical research  
- Mental health analytics  

---

##  Contributors

- **Tanishq Katoch** – HRV feature extraction, model development   
- **Sampriti Mohanty** – Data preprocessing,  Testing  
- **Shreyansh Gaur** – Visualization, Evaluation  

This project was developed collaboratively as part of an academic assignment.

Note: This repository represents the individual submission of the same collaborative project.

---

##  Future Improvements

- Deep learning-based HRV modeling  
- Larger dataset validation  
- Real-time wearable integration  
- Deployment as a web/mobile application  

---

##  License

This project is developed for academic and educational purposes.
