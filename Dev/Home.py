



# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import pandas as pd
# import numpy as np
# import uvicorn
# import os

# # call the app
# app = FastAPI(title="Sepsis Prediction API")

# # Load the models and scaler
# def load_models_and_scaler():
#     with open("Assets/models/gradient_boosting_classifier.pkl", "rb") as f1, open("Assets/models/random_forest_classifier.pkl", "rb") as f2, open("Assets/models/scaler.pkl", "rb") as f3:
#         gbc_model = pickle.load(f1)
#         rf_model = pickle.load(f2)
#         scaler = pickle.load(f3)
#     return gbc_model, rf_model, scaler

# gbc_model, rf_model, scaler = load_models_and_scaler()

# def predict(df, model):
#     # Scaling
#     scaled_df = scaler.transform(df)

#     # Prediction
#     prediction = model.predict_proba(scaled_df)
#     highest_proba = prediction.max(axis=1)

#     # Generate response
#     predicted_labels = ["Patient does not have sepsis" if proba <= 0.5 else "Patient has sepsis" for proba in highest_proba]
#     response = generate_response(predicted_labels, highest_proba)
#     return response

# def generate_response(predicted_labels, probabilities):
#     response = []
#     for label, proba in zip(predicted_labels, probabilities):
#         output = {
#             "prediction": label,
#             "probability of prediction": str(round(proba * 100)) + '%'
#         }
#         response.append(output)
#     return response

# class Patient(BaseModel):
#     Plasma_glucose: int
#     Blood_Work_R1: int
#     Blood_Pressure: int
#     Blood_Work_R3: int
#     BMI: float
#     Blood_Work_R4: float
#     Patient_age: int
   

# class Patients(BaseModel):
#     all_patients: list[Patient]

#     @classmethod
#     def return_list_of_dict(cls, patients: "Patients"):
#         patient_list = []
#         for patient in patients.all_patients:
#             patient_dict = patient.dict()
#             patient_list.append(patient_dict)
#         return patient_list
    
# # Endpoints
# # Root Endpoint
# @app.get("/")
# def home():
#     return {
#         "message": "Welcome to the Sepsis Prediction API!",
#         "description": "This API provides endpoints for predicting sepsis in patients.",
#         "endpoints": {
#             "/predict_gbc": {
#                 "method": "POST",
#                 "description": "Predict sepsis using Gradient Boosting Classifier for a single patient."
#             },
#             "/predict_rf": {
#                 "method": "POST",
#                 "description": "Predict sepsis using Random Forest Classifier for a single patient."
#             },
#             "/predict_multiple": {
#                 "method": "POST",
#                 "description": "Predict sepsis for multiple patients using both models."
#             },
#             "/documents": {
#                 "method": "GET",
#                 "description": "Get documentation for this API."
#             }
#         }
#     }

# # Prediction endpoint for Gradient Boosting Classifier
# @app.post("/predict_gbc")
# def predict_sepsis_gbc(patient: Patient):
#     # Make prediction using Gradient Boosting Classifier
#     data = pd.DataFrame(patient.dict(), index=[0])
#     gbc_parsed = predict(data, gbc_model)
#     return {"output": gbc_parsed}

# # Prediction endpoint for Random Forest Classifier
# @app.post("/predict_rf")
# def predict_sepsis_rf(patient: Patient):
#     # Make prediction using Random Forest Classifier
#     data = pd.DataFrame(patient.dict(), index=[0])
#     rf_parsed = predict(data, rf_model)
#     return {"output": rf_parsed}

# # Multiple Prediction Endpoint
# @app.post("/predict_multiple")
# def predict_sepsis_for_multiple_patients(patients: Patients):
#     """Make prediction with the passed data"""
#     data = pd.DataFrame(Patients.return_list_of_dict(patients))
#     gbc_parsed = predict(data, gbc_model)
#     rf_parsed = predict(data, rf_model)
#     return {"output_gbc": gbc_parsed, "output_rf": rf_parsed}

# # Documentation Endpoint
# @app.get('/documents')
# def documentation():
#     return{'description':'All documentation'}

# if __name__ == "__main__":
#     uvicorn.run("main:app", reload=True)



from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import uvicorn
import os

# call the app
app = FastAPI(title="Sepsis Prediction With FastAPI")

# Load the models and scaler
def load_models_and_scaler():
    with open("Assets/models/gradient_boosting_classifier.pkl", "rb") as f1, open("Assets/models/random_forest_classifier.pkl", "rb") as f2, open("Assets/models/scaler.pkl", "rb") as f3:
        gbc_model = pickle.load(f1)
        rf_model = pickle.load(f2)
        scaler = pickle.load(f3)
    return gbc_model, rf_model, scaler

gbc_model, rf_model, scaler = load_models_and_scaler()

def predict(df, model):
    # Scaling
    scaled_df = scaler.transform(df)

    # Prediction
    prediction = model.predict_proba(scaled_df)
    highest_proba = prediction.max(axis=1)

    # Generate response
    predicted_labels = ["Patient does not have sepsis" if proba <= 0.5 else "Patient has sepsis" for proba in highest_proba]
    response = generate_response(predicted_labels, highest_proba)
    return response

def generate_response(predicted_labels, probabilities):
    response = []
    for label, proba in zip(predicted_labels, probabilities):
        output = {
            "prediction": label,
            "probability of prediction": str(round(proba * 100)) + '%'
        }
        response.append(output)
    return response

class Patient(BaseModel):
    Plasma_glucose: int
    Blood_Work_R1: int
    Blood_Pressure: int
    Blood_Work_R3: int
    BMI: float
    Blood_Work_R4: float
    Patient_age: int
   

class Patients(BaseModel):
    all_patients: list[Patient]

    @classmethod
    def return_list_of_dict(cls, patients: "Patients"):
        patient_list = []
        for patient in patients.all_patients:
            patient_dict = patient.dict()
            patient_list.append(patient_dict)
        return patient_list
    
# Endpoints
# Root Endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to the Sepsis Prediction API!",
        "description": "This API provides endpoints for predicting sepsis in patients.",
        "usage_instructions": {
            "predict_gbc": {
                "method": "POST",
                "description": "Predict sepsis using Gradient Boosting Classifier for a single patient.",
                "body": "Send a POST request with JSON data representing a single patient's information. Example: {'Plasma_glucose': 120, 'Blood_Work_R1': 80, ...}"
            },
            "predict_rf": {
                "method": "POST",
                "description": "Predict sepsis using Random Forest Classifier for a single patient.",
                "body": "Send a POST request with JSON data representing a single patient's information. Example: {'Plasma_glucose': 120, 'Blood_Work_R1': 80, ...}"
            },
            "predict_multiple": {
                "method": "POST",
                "description": "Predict sepsis for multiple patients using both models.",
                "body": "Send a POST request with JSON data representing a list of patients' information. Example: {'all_patients': [{'Plasma_glucose': 120, 'Blood_Work_R1': 80, ...}, {'Plasma_glucose': 130, 'Blood_Work_R1': 85, ...}]}"
            },
            "/documents": {
                "method": "GET",
                "description": "Get documentation for this API."
            }
        }
    }

# Prediction endpoint for Gradient Boosting Classifier
@app.post("/predict_gbc")
def predict_sepsis_gbc(patient: Patient):
    # Make prediction using Gradient Boosting Classifier
    data = pd.DataFrame(patient.dict(), index=[0])
    gbc_parsed = predict(data, gbc_model)
    return {"output": gbc_parsed}

# Prediction endpoint for Random Forest Classifier
@app.post("/predict_rf")
def predict_sepsis_rf(patient: Patient):
    # Make prediction using Random Forest Classifier
    data = pd.DataFrame(patient.dict(), index=[0])
    rf_parsed = predict(data, rf_model)
    return {"output": rf_parsed}

# Multiple Prediction Endpoint
@app.post("/predict_multiple")
def predict_sepsis_for_multiple_patients(patients: Patients):
    """Make prediction with the passed data"""
    data = pd.DataFrame(Patients.return_list_of_dict(patients))
    gbc_parsed = predict(data, gbc_model)
    rf_parsed = predict(data, rf_model)
    return {"output_gbc": gbc_parsed, "output_rf": rf_parsed}

# Documentation Endpoint
@app.get('/documents')
def documentation():
    return{'description':'All documentation'}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
