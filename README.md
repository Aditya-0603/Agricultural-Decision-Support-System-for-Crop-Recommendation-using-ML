# Agricultural Decision Support System for Crop Recommendation

A Machine Learning-based crop recommendation system that helps farmers and agricultural stakeholders identify the most suitable crop based on soil nutrients and environmental conditions. The system analyzes key agricultural parameters and provides real-time crop recommendations through an interactive Streamlit web application.

## 📌 Project Overview

This project leverages Machine Learning algorithms to recommend the best crop for cultivation based on:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH Value
- Rainfall

The model was trained on agricultural data containing 2,200 records across 22 crop categories.

---

## 🚀 Features

- Crop recommendation based on soil and climate conditions
- Comparison of multiple Machine Learning algorithms
- Real-time prediction through Streamlit web interface
- Data visualization and exploratory data analysis (EDA)
- High-accuracy crop prediction system

---

## 📊 Dataset Information

| Attribute | Value |
|------------|--------|
| Total Records | 2,200 |
| Features | 7 |
| Crop Categories | 22 |
| Problem Type | Multi-Class Classification |

### Input Features

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

### Target Variable

- Crop Label

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 🤖 Machine Learning Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## 📈 Model Performance

After evaluating multiple classification models, the Random Forest Classifier achieved the best performance.

| Model | Accuracy |
|---------|---------|
| Logistic Regression | Evaluated |
| KNN | Evaluated |
| Decision Tree | Evaluated |
| Random Forest | ~99% |
| SVM | Evaluated |

**Best Model:** Random Forest Classifier

**Accuracy Achieved:** ~99%

---

## 📊 Exploratory Data Analysis

Performed:

- Correlation Analysis
- Feature Distribution Analysis
- Histograms
- Heatmaps
- Data Visualization
- Feature Scaling

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit web application where users can:

1. Enter soil nutrient values.
2. Enter weather-related parameters.
3. Get instant crop recommendations.
4. View predicted crop output in real time.

Run the application:

```bash
streamlit run app.py
```

---

## 📷 Sample Prediction Workflow
<img width="1660" height="921" alt="image" src="https://github.com/user-attachments/assets/f7fc015f-db46-4191-8a9b-9503faf0cdf0" />


1. Enter:
   - N = 90
   - P = 42
   - K = 43
   - Temperature = 20°C
   - Humidity = 82%
   - pH = 6.5
   - Rainfall = 202 mm

2. Click **Predict**

3. Receive recommended crop instantly.

---

## 🎯 Future Improvements

- Crop yield prediction
- Fertilizer recommendation system
- Soil health analysis
- Weather API integration
- Deployment on cloud platforms
- Mobile application support

---

## 📚 Learning Outcomes

- Data Preprocessing
- Feature Engineering
- Classification Algorithms
- Model Evaluation
- Streamlit Deployment
- Agricultural Data Analytics

---

## 👨‍💻 Author

**Aditya Shivaji Warungase**

- Python
- Machine Learning
- SQL
- Data Analytics
- Streamlit

---

⭐ If you found this project useful, consider giving it a star.
