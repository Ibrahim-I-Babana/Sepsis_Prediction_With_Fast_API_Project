# from fastapi import FastAPI
# import uvicorn
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import numpy as np

# app = FastAPI()

# class input_features(BaseModel):
#     Plasma_glucose: int
#     Blood_Work_R1: int
#     Blood_Pressure: int
#     Blood_Work_R2: int
#     Blood_Work_R3: int
#     BMI: float
#     Blood_Work_R4: float
#     Patient_age: int
#     Insurance: int


# forest_pipeline = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_random_forest_model.joblib")
# encoder = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_pipeline_rf.joblib") 

# @app.get('/')
# def home_page(Name=None):
#     return {"message":"Hello World!"}

# @app.post('/predict_random_forest')
# def predict(data:input_features):
#     def random_forest_prediction(data:input_features):
#         #prediction = forest_pipeline.predict([data.Plasma_glucose, data.Blood_Work_R1, data.Blood_Pressure, data.Blood_Work_R2, data.Blood_Work_R3, data.Blood_Work_R4, data.BMI, data.Patient_age, data.Insurance])
#         # Convert model to a DataFrame
#         df = pd.DataFrame([data.model_dump()])
        
#         # Make predictions
#         prediction = forest_pipeline.predict(df)
#         return {'Prediction': prediction}
    
#     #     # Decode Sepsis using LabelEncoder
#     # decoded_sepsis_rf = label_encoder.inverse_transform([rf_prediction])[0]

#     return random_forest_prediction(data)

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)



# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle
# import pandas as pd
# import numpy as np
# import uvicorn
# import os

# # call the app
# app = FastAPI(title="API")

# # Load the model and scaler
# def load_model_and_scaler():
#     with open("model.pkl", "rb") as f1, open("scaler.pkl", "rb") as f2:
#         return pickle.load(f1), pickle.load(f2)

# model, scaler = load_model_and_scaler()

# def predict(df, endpoint="simple"):
#     # Scaling
#     scaled_df = scaler.transform(df)

#     # Prediction
#     prediction = model.predict_proba(scaled_df)

#     highest_proba = prediction.max(axis=1)

#     predicted_labels = ["Patient does not have sepsis" if i == 0 else f"Patient has sepsis" for i in highest_proba]
#     print(f"Predicted labels: {predicted_labels}")
#     print(highest_proba)

#     response = []
#     for label, proba in zip(predicted_labels, highest_proba):
#         output = {
#             "prediction": label,
#             "probability of prediction": str(round(proba * 100)) + '%'
#         }
#         response.append(output)

#     return response


# class Patient(BaseModel):
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
# def root():
#     return {"API": "This is an API for sepsis prediction."}

# # Prediction endpoint
# @app.post("/predict")
# def predict_sepsis(patient: Patient):
#     # Make prediction
#     data = pd.DataFrame(patient.dict(), index=[0])
#     parsed = predict(df=data)
#     return {"output": parsed}

# # Multiple Prediction Endpoint
# @app.post("/predict_multiple")
# def predict_sepsis_for_multiple_patients(patients: Patients):
#     """Make prediction with the passed data"""
#     data = pd.DataFrame(Patients.return_list_of_dict(patients))
#     parsed = predict(df=data, endpoint="multi")
#     return {"output": parsed}

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
app = FastAPI(title="API")

# Load the models and scaler
def load_models_and_scaler():
    with open("models/gradient_boosting_classifier.pkl", "rb") as f1, open("models/random_forest_classifier.pkl", "rb") as f2, open("models/scaler.pkl", "rb") as f3:
        gbc_model = pickle.load(f1)
        rf_model = pickle.load(f2)
        scaler = pickle.load(f3)
    return gbc_model, rf_model, scaler

gbc_model, rf_model, scaler = load_models_and_scaler()

def predict(df):
    # Scaling
    scaled_df = scaler.transform(df)

    # Prediction
    gbc_prediction = gbc_model.predict_proba(scaled_df)
    rf_prediction = rf_model.predict_proba(scaled_df)

    gbc_highest_proba = gbc_prediction.max(axis=1)
    rf_highest_proba = rf_prediction.max(axis=1)

    # Combine predictions from both models
    combined_prediction = (gbc_highest_proba + rf_highest_proba) / 2

    predicted_labels = ["Patient does not have sepsis" if proba <= 0.5 else "Patient has sepsis" for proba in combined_prediction]

    response = []
    for label, proba in zip(predicted_labels, combined_prediction):
        output = {
            "prediction": label,
            "probability of prediction": str(round(proba * 100)) + '%'
        }
        response.append(output)

    return response


class Patient(BaseModel):
    Plasma_glucose:int
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
def root():
    return {"API": "This is an API for sepsis prediction."}

# Prediction endpoint
@app.post("/predict")
def predict_sepsis(patient: Patient):
    # Make prediction
    data = pd.DataFrame(patient.dict(), index=[0])
    parsed = predict(df=data)
    return {"output": parsed}

# Multiple Prediction Endpoint
@app.post("/predict_multiple")
def predict_sepsis_for_multiple_patients(patients: Patients):
    """Make prediction with the passed data"""
    data = pd.DataFrame(Patients.return_list_of_dict(patients))
    parsed = predict(df=data)
    return {"output": parsed}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
