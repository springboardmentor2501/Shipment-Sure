# 📘 Milestone 4 – Week 7–8
Model Deployment & Final Documentation

## 🟦 1. Overview

Milestone 4 focuses on deploying the best performing model from Milestone 3 and building a simple web application that predicts whether a shipment will be delivered on time or delayed.

This milestone also includes preparing the final project documentation, report, and presentation.

## 🟦 2. Objective of Milestone 4

- streamlit run app/streamlit_app.py
- eploy the selected ML model (XGBoost)
- Build a user interface where users can input shipment details
- Display prediction results in real-time
- Finalize documentation and prepare a presentable project

## 🟦 3. Deployment Approach

You will use:

### ✔ Streamlit (Recommended – simple, fast, modern)

OR

### ✔ Flask (If required by mentor)

We choose Streamlit because:

- No frontend knowledge required
- Easy to deploy
- Quick to integrate with ML models
- Perfect for internship-level projects

## 🟦 4. Saved Model Used for Deployment

From Milestone 3:

- File: best_model.pkl
- Model: XGBoost Classifie
- Stored in project root or model/ folder

This model is loaded inside Streamlit for prediction.

## 🟦 5. Application Workflow

```pgsql
User Inputs Features
↓
Preprocessing (same steps as training)
↓
Load Best Model
↓
Prediction (On time / Delayed)
↓
Display Result on UI
```
<img width="2719" height="875" alt="Application Workflow" src="https://github.com/user-attachments/assets/91f79e9e-3561-4146-8283-49aa2e3859d9" />

## 🟦 6. Streamlit App Structure

The app file is:

```bash
app/streamlit_app.py
```

Inside it:

- Load model (joblib.load)
- Create input fields (select boxes + sliders + numeric inputs)
- Preprocess input
- Generate prediction
- Display output

## 🟦 7. Key Features of the App
### ✔ Input fields for:

<img width="1917" height="967" alt="Prediction-page" src="https://github.com/user-attachments/assets/10540953-23a0-478a-9204-70adad7fcc7a" />
- Shipping distance
- Delivery days
- Supplier rating
- Shipment mode
- Engineered features

### ✔ On-click prediction

- Model predicts:
  - On-Time Delivery
  - Delayed Delivery

<img width="1905" height="957" alt="Prediction-page-output" src="https://github.com/user-attachments/assets/b2a528eb-af78-437d-8393-63f28b326b0d" />
### ✔ Probability Score

- shows confidence of prediction (0–100%)

### ✔ Clean & simple UI

- Title
- Description
- Inputs arranged in columns
- Button-trigger prediction

<img width="1918" height="951" alt="Model-info" src="https://github.com/user-attachments/assets/4a851c6e-6417-4c07-93e5-50caf91f66bb" />

<img width="1920" height="951" alt="EDA-preview-1" src="https://github.com/user-attachments/assets/6a62808a-8b09-4d22-916c-f43761accc10" />

<img width="1917" height="958" alt="EDA-preview-2" src="https://github.com/user-attachments/assets/88821da1-69fe-4a4e-acfe-8b645eb4220a" />
## 🟦 8. Running the Application

From terminal:

```arduino
streamlit run app/streamlit_app.py
```

## 🟦 9. Final Documentation & Submission

Milestone 4 includes preparing:

### ✔ GitHub Repository

Contains:
- Data folder
- Scripts folder
- App folder
- Notebooks (EDA + Modeling)
- Docs folder (all milestone reports)
- README.md
- requirements.txt

### ✔ Final PDF Report

Should include:
- Objectives
- Dataset
- Pipeline
- EDA
- Models
- Evaluation
- Deployment screenshots
- Conclusion

### ✔ Presentation (PPTX)

Slides include:
- Project overview
- Steps
- Insights
- Model results
- App demo
- Conclusion

## 🟦 10. Deliverables for Milestone 4
| Deliverable | Status |
|-------------|--------|
| Streamlit/Flask app | ✔ Completed |
| App uses loaded ML model | ✔ Completed |
| Prediction UI | ✔ Completed |
| GitHub repository updated | ✔ Completed |
| Final report (PDF) | ✔ Completed |
| Presentation (PPTX) | ✔ Completed |

## 🟦 11. Summary

Milestone 4 successfully deployed the trained XGBoost model into a Streamlit web interface.
Users can now input shipment details and receive real-time predictions.
All documentation, reports, and the GitHub repository are finalized, completing the project end-to-end.
