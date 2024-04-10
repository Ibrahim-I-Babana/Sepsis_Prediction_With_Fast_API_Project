# from fastapi import FastAPI
# import uvicorn
# from pydantic import BaseModel
# import joblib
# import pandas as pd


# app = FastAPI()

# class input_features(BaseModel):
#     Plasma_glucose: object
#     Blood_Work_R1: object
#     Blood_Pressure: object
#     Blood_Work_R2: object
#     Blood_Work_R3: object
#     BMI: object
#     Blood_Work_R4: object
#     Patient_age: object
#     Insurance: object


# forest_pipeline = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_random_forest_model.joblib")
# encoder = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_pipeline.joblib") 

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
    
#     return random_forest_prediction(data)

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)





# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# import os
# import uvicorn
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import LabelEncoder

# app = FastAPI()

# class InputFeatures(BaseModel):
#     Plasma_glucose: object
#     Blood_Work_R1: object
#     Blood_Pressure: object
#     Blood_Work_R2: object
#     Blood_Work_R3: object
#     BMI: object
#     Blood_Work_R4: object
#     Patient_age: object
#     Insurance: object

# # Load the trained model and encoder
# forest_pipeline = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_random_forest_model.joblib")
# encoder = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_pipeline.joblib")

# def load_and_preprocess_data():
#     # Load the dataset
#     current_dir = os.getcwd()
#     train_data_path = os.path.join(current_dir, 'Datasets', 'Paitients_Files_Train.csv')
#     train_data = pd.read_csv(train_data_path)

#     # Separate features and target
#     X = train_data.drop('Sepssis', axis=1)
#     y = train_data['Sepssis']

#     # Identify numeric and categorical columns
#     numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
#     categorical_cols = X.select_dtypes(include=['object']).columns

#     # Impute missing values for numeric columns
#     imputer = SimpleImputer(strategy='mean')
#     X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

#     # Convert categorical columns to numeric using label encoding
#     le = LabelEncoder()
#     X[categorical_cols] = X[categorical_cols].apply(lambda col: le.fit_transform(col.astype(str)))

#     return X, y

# # Load and preprocess data
# X_train, y_train = load_and_preprocess_data()

# # Fit the RandomForestClassifier model
# forest_pipeline = RandomForestClassifier()
# forest_pipeline.fit(X_train, y_train)

# @app.get('/')
# def home_page(Name=None):
#     return {"message": "Hello World!"}

# @app.post('/predict_random_forest')
# def predict(data: InputFeatures):
#     def random_forest_prediction(data: InputFeatures):
#         # Convert input data to DataFrame
#         df = pd.DataFrame(data.dict(), index=[0])
        
#         # Make predictions
#         prediction = forest_pipeline.predict(df)
        
#         # Convert prediction to an int instead of an array
#         prediction = int(prediction[0])

#         # Decode using our encoder
#         prediction = encoder.inverse_transform([prediction])[0]
        
#         return {'Prediction': prediction}

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000)




# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from typing import List

# # Define FastAPI app
# app = FastAPI()

# # Define input features
# input_features = ['Plasma_glucose', 'Blood_Work_R1', 'Blood_Pressure', 
#                   'Blood_Work_R2', 'Blood_Work_R3', 'BMI', 
#                   'Blood_Work_R4', 'Patient_age', 'Insurance']

# # Load the trained models and preprocessor
# best_gb_model = joblib.load("models/best_gradient_boosting_model.joblib")
# pipeline = joblib.load("models/best_pipeline.joblib")
# best_rf_model = joblib.load("models/best_random_forest_model.joblib")
# best_cb_model = joblib.load("models/best_catboosting_model.joblib")

# # Define input data model
# class InputData(BaseModel):
#     Plasma_glucose: float
#     Blood_Work_R1: float
#     Blood_Pressure: float
#     Blood_Work_R2: float
#     Blood_Work_R3: float
#     BMI: float
#     Blood_Work_R4: float
#     Patient_age: int
#     Insurance: str

# # Define output data model
# class OutputData(BaseModel):
#     Sepsis: str

# # Define endpoint to make predictions using Gradient Boosting model
# @app.post("/predict_gb", response_model=OutputData)
# def predict_gb(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     preprocessed_data = pipeline.transform(input_df)
#     prediction = best_gb_model.predict(preprocessed_data)
#     return {"Sepsis": str(prediction[0])}

# # Define endpoint to make predictions using Random Forest model
# @app.post("/predict_rf", response_model=OutputData)
# def predict_rf(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     preprocessed_data = pipeline.transform(input_df)
#     prediction = best_rf_model.predict(preprocessed_data)
#     return {"Sepsis": str(prediction[0])}

# # Define endpoint to make predictions using CatBoost model
# @app.post("/predict_cb", response_model=OutputData)
# def predict_cb(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     preprocessed_data = pipeline.transform(input_df)
#     prediction = best_cb_model.predict(preprocessed_data)
#     return {"Sepsis": str(prediction[0])}

# # Define a health check endpoint
# @app.get("/health")
# def health_check():
#     return {"status": "Healthy"}

# # Dockerization instructions
# # Dockerfile
# """
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# COPY ./app /app
# """

# # Command to build Docker image
# # docker build -t sepsis-prediction-app .

# # Command to run Docker container
# # docker run -d --name sepsis-prediction-container -p 80:80 sepsis-prediction-app







# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from typing import List

# # Define FastAPI app
# app = FastAPI()

# # Define input features
# input_features = ['Plasma_glucose', 'Blood_Work_R1', 'Blood_Pressure', 
#                   'Blood_Work_R2', 'Blood_Work_R3', 'BMI', 
#                   'Blood_Work_R4', 'Patient_age', 'Insurance']

# @app.get('/')
# def home_page(Name=None):
#     return {"message":"Hello World!"}

# # Load the trained models and preprocessor
# best_gb_model = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_gradient_boosting_model.joblib")
# pipeline = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_pipeline.joblib")
# best_rf_model = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_random_forest_model.joblib")

# # Define input data model
# class InputData(BaseModel):
#     Plasma_glucose: float
#     Blood_Work_R1: float
#     Blood_Pressure: float
#     Blood_Work_R2: float
#     Blood_Work_R3: float
#     BMI: float
#     Blood_Work_R4: float
#     Patient_age: int
#     Insurance: str

# # Define output data model
# class OutputData(BaseModel):
#     Sepsis: str

# # Define endpoint to make predictions using Gradient Boosting model
# @app.post("/predict_gb", response_model=OutputData)
# def predict_gb(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     preprocessed_data = pipeline.transform(input_df)
#     prediction = best_gb_model.predict(preprocessed_data)
#     return {"Sepsis": str(prediction[0])}

# # Define endpoint to make predictions using Random Forest model
# @app.post("/predict_rf", response_model=OutputData)
# def predict_rf(data: InputData):
#     input_df = pd.DataFrame([data.dict()])
#     preprocessed_data = pipeline.transform(input_df)
#     prediction = best_rf_model.predict(preprocessed_data)
#     return {"Sepsis": str(prediction[0])}

# # Define a health check endpoint
# @app.get("/health")
# def health_check():
#     return {"status": "Healthy"}

# # Dockerization instructions
# # Dockerfile
# """
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# COPY ./app /app
# """

# # Command to build Docker image
# # docker build -t sepsis-prediction-app .

# # Command to run Docker container
# # docker run -d --name sepsis-prediction-container -p 80:80 sepsis-prediction-app




# import joblib
# import numpy as np
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.base import TransformerMixin
# from notebook import SkewnessTransformer

# app = FastAPI()

# # Load trained models and label encoder
# random_forest_model = joblib.load("Models/random_forest_model.pkl")
# logistic_regression_model = joblib.load("Models/decision_tree_model.pkl")
# label_encoder = joblib.load("Models/label_encoder.pkl")
# scaler = joblib.load("Models/preprocessor.pkl")  # Assuming you have saved RobustScaler

# class Features(BaseModel):
#     PRG: int
#     PL: int
#     PR: int
#     SK: int
#     TS: int
#     M11: float
#     BD2: float
#     Age: int
#     Insurance: int

# @app.post("/predict_sepsis")
# async def predict_sepsis(features: Features):
#     # Convert features to numpy array
#     feature_values = np.array([[
#         features.PRG, features.PL, features.PR, features.SK, features.TS,
#         features.M11, features.BD2, features.Age, features.Insurance
#     ]])

#     # Normalize features using RobustScaler
#     normalized_features = scaler.transform(feature_values)

#     # Predict Sepsis using the Random Forest model
#     rf_prediction = random_forest_model.predict(normalized_features)[0]

#     # Predict Sepsis using the Logistic Regression model
#     dt_prediction = logistic_regression_model.predict(normalized_features)[0]

#     # Decode Sepsis using LabelEncoder
#     decoded_sepsis_rf = label_encoder.inverse_transform([rf_prediction])[0]
#     decoded_sepsis_dt = label_encoder.inverse_transform([dt_prediction])[0]

#     return {
#         "Random Forest Prediction": decoded_sepsis_rf,
#         "Random Tree Prediction": decoded_sepsis_dt
#     }



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List

# Define FastAPI app
app = FastAPI()

# Define input features
input_features = ['Plasma_glucose', 'Blood_Work_R1', 'Blood_Pressure', 
                  'Blood_Work_R2', 'Blood_Work_R3', 'BMI', 
                  'Blood_Work_R4', 'Patient_age', 'Insurance']

@app.get('/')
def home_page(Name=None):
    return {"message":"Hello World!"}

# Load the trained models and preprocessor
best_gb_model = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_gradient_boosting_model.joblib")
pipeline = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_pipeline.joblib")
best_rf_model = joblib.load("C:/Users/user/OneDrive/Desktop/MY DS CAREER ACCELERATOR/Sepsis_Prediction_With_Fast_API_Project/models/best_random_forest_model.joblib")



# Define input data model
class InputData(BaseModel):
    Plasma_glucose: float
    Blood_Work_R1: float
    Blood_Pressure: float
    Blood_Work_R2: float
    Blood_Work_R3: float
    BMI: float
    Blood_Work_R4: float
    Patient_age: int
    Insurance: str

# Define output data model
class OutputData(BaseModel):
    Sepsis: str

# Define endpoint to make predictions using Gradient Boosting model
@app.post("/predict_gb", response_model=OutputData)
def predict_gb(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = best_gb_model.predict(input_df)
    return {"Sepsis": str(prediction[0])}

# Define endpoint to make predictions using Random Forest model
@app.post("/predict_rf", response_model=OutputData)
def predict_rf(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = best_rf_model.predict(input_df)
    return {"Sepsis": str(prediction[0])}

# Define a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "Healthy"}
