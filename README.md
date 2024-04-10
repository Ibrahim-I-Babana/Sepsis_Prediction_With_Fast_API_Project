### **📊 Sepsis Prediction with FastAPI 🚀** ###

**Welcome to the Sepsis Prediction with FastAPI Project! Sepsis is a life-threatening medical condition characterized by a systemic inflammatory response to infection, leading to organ dysfunction and failure. Early detection of sepsis is crucial for timely intervention and improved patient outcomes. This project aims to develop a predictive model for sepsis detection in patients within the Intensive Care Unit (ICU) using various input features such as plasma glucose level, blood work results, blood pressure, body mass index, patient's age, and insurance status.**


### **Business Understanding** ###

**The primary objective is to develop a predictive model that assists healthcare professionals in assessing the likelihood of sepsis development based on key clinical indicators. The model's predictions will enable healthcare providers to identify patients at higher risk of sepsis, facilitating prompt and targeted medical interventions.**

**Analyzing the distribution of different features between patients with and without sepsis can have several business implications in a healthcare setting:**

**1. Identification of risk factors**

**2. Targeted intervention strategies**

**3. Resource allocation**

**4. Quality improvement initiatives**

**5. Cost management**


### **Description of Columns** ##

**ID:** *Unique patient identifier*

**PRG:** *Plasma glucose level*

**PL:** *Blood Work Result-1 (mu U/ml)*

**PR:** *Blood Pressure (mm Hg)*

**SK:** *Blood Work Result-2 (mm)*

**TS:** *Blood Work Result-3 (mu U/ml)*

**M11:** *Body mass index (weight in kg/(height in m)^2)*

**BD2:** *Blood Work Result-4 (mu U/ml)*

**Age:** *Patient's age (years)*

**Insurance:** *Insurance coverage status*

**Sepssis:** *Target variable indicating sepsis occurrence (Positive/Negative)*

### **Analytical Questions** ###

**1.** *How does Plasma Glucose vary between patients with and without sepsis?*

**2.** *Are there Patterns or Thresholds in Blood Work Results Associated with Sepsis?*

**3.** *How does the distribution of Blood Pressure vary between patients with and without sepsis?*

**4.** *Are there any differences in the distribution of Insurance coverage between patients with and without sepsis?*

**5.** *What is the distribution of Body Mass Index (BMI) for patients with and without sepsis?*

**6.** *How does the age distribution differ between patients with and without sepsis?*

**7.** *Are certain combinations of variables more indicative of sepsis risk?*

**8** *What are the key factors contributing to the model's performance and limitations?*

### **Objectives** ###

**1.** *Investigate the distribution of plasma glucose levels among patients with and without sepsis.*

**2.** *Compare the distribution of blood work results for patients with and without sepsis.*

**3.** *Examine the relationship between blood pressure values and the occurrence of sepsis.*

**4.** *Analyze the distribution of Body Mass Index among individuals with and without sepsis.*

**5.** *Explore the age distribution of patients with sepsis compared to those without.*

**6.** *Determine the percentage of patients with valid insurance among those with and without sepsis.*

**7.** *Assess how well combined factors predict sepsis occurrence.*

**8.** *Evaluate the accuracy of the predictive model in identifying sepsis based on selected features.*

### **Project Structure** ###

Sales-Prediction-Exploration
├── data
│ ├── train.csv
│ ├── test.csv
├── notebook: codebabana 
│ ├── 01_Exploratory_Data_Analysis.ipynb
│ ├── 02_Preprocessing_and_Feature_Engineering.ipynb
│ ├── 03_Model_Application.ipynb
│ └── ...
├── README.md
└── requirements.txt

The `Datasets` folder encompasses essential datasets for our analysis:
- `train.csv`: Training data with Sepsis information.
- `test.csv`: Test data for predicting Sepsis.


### **Notebook** ###

**Explore detailed analyses in the 'notebook':**

- `01_Exploratory_Data_Analysis: In-depth exploration of Sepsis data.
- `02_Preprocessing_and_Feature_Engineering: Preprocessing steps and feature engineering.
- `03_Model_Application: Application of One-Hot Encoding, Random Forests, GradientBoosting, Hyperparameter Tuning and ROC AUC Scoring.


### **Requirements** ###

**The necessary Python packages are outlined in** `requirements.txt`.

### **Hypothesis** ###

**Null Hypothesis (H0):**

*There is no significant difference in Blood Pressure and BMI between patients with and without sepsis.*

**Alternative Hypothesis (H1):**

*There is a significant difference in Blood Pressure and BMI between patients with and without sepsis.*

### **Data Loading** ###

**The dataset consists of training and testing data containing various features and the target variable indicating sepsis occurrence. The data will be loaded for analysis and model development.**

### **Key Insights Identified in the Dataset and Potential Solutions** ###

**Data Types:** *Addressing data types and potential categorical encoding.*

**Outliers:** *Handling outliers in numerical columns.*

**Missing Values:** *Verifying and addressing any missing values.*

**Duplicates:** *Checking and removing duplicate rows.*

**Categorical Data Encoding:** *Encoding categorical data.*

**Column Names:** *Renaming columns for better clarity.*

**Data Distribution:** *Visualizing feature distributions.*

**Scaling:** *Scaling numerical features for model fitting.*

**Data Interpretation:** *Considering domain knowledge for interpretation.*

### **Initial Model Evaluation Results** ###

**Various classification models such as Logistic Regression, Random Forest, Gaussian Naive Bayes, Gradient Boosting, XGBoost, and LightGBM were evaluated using accuracy, F1 score, ROC AUC score, precision, and recall. Random Forest and Gradient Boosting showed promising performance based on initial evaluation.**

### **Hyperparameter Tuning Results** ###

**Hyperparameter tuning was performed for each model to optimize performance. The best parameters and corresponding scores were recorded, with Random Forest achieving the highest ROC AUC score.**

### **ROC AUC and Best Scores of the Models** ###

**Random Forest, Gradient Boosting, and LightGBM showed the highest ROC AUC scores among the models evaluated.**

### **Conclusion** ###

**The README provides a comprehensive overview of the project, including its objectives, analytical questions, hypothesis, data description, key insights, and initial model evaluation results. It serves as a guide for understanding the project's goals, methodologies, and findings, facilitating collaboration and knowledge sharing among stakeholders**
