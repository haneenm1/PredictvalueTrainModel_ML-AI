# README: PPV Prediction using XGBoost

## Project Overview
This project aims to predict the PPV using the given dataset that includes Gender, Age, and Dur. The solution uses the XGBoost algorithm for building a regression model.

## Steps in the Process
The project follows the Data Science Life Cycle:
1. Define and Understand the Problem:
   - The task is to predict PPV, a regression problem, using features like Age, Gender, and Dur.
   - XGBoost was chosen for its robustness and performance.

2. Data Collection (Given):
   - **Load the Dataset:** Read the provided Excel file to understand its structure and contents.
   - **Inspect Columns:** Focus on columns [Gender, Age, Dur, PPV]. Ensure all columns are clean and free of missing values.

3. Data Cleaning and Preparation:
   - Handle missing or invalid data.
   - Perform encoding on the Gender column to convert it into numeric values.
   - Split the data into training and testing sets (80:20 ratio).

4. Exploratory Data Analysis (EDA):
   - Analyze feature distributions using histograms.
   - Identify relationships between features and the target variable using scatter plots.

5. Model Building and Training:
   - The XGBoost Regressor is trained on the prepared data.

6. Model Evaluation:
   - Metrics used to evaluate model performance:
     - **RMSE (Root Mean Squared Error):** Measures prediction error magnitude.
     - **MAE (Mean Absolute Error):** Measures absolute differences between predictions and actual values.

7. Model Saving:
   - The trained XGBoost model is saved as `xgboost_model.pkl` for future use.

## How to Run the Code
### Prerequisites:
- Jupyter Notebook.
- Required libraries: pandas, numpy, matplotlib, seaborn, xgboost, scikit-learn, joblib.
- Install missing libraries using:
  ```bash
  pip install pandas numpy matplotlib seaborn xgboost scikit-learn joblib
  ```

### Steps to Run:
1. **Prepare the Dataset:**
   - Save the dataset as a CSV file named `data.csv` in the working directory.
2. **Run the Jupyter Notebook:**
   - Open the file in Jupyter Notebook: `ppv_prediction.ipynb`.
3. **Model Training:**
   - Follow the steps to preprocess the data, build, and evaluate the model.

## Results:
- RMSE, MAE, and R² (Accuracy) values will be printed.
- Scatter plot comparing Actual vs Predicted values will be displayed.

## Justification for XGBoost
### Why XGBoost?
- XGBoost is a gradient-boosting algorithm that is optimized for speed and performance.
- It handles missing values efficiently and works well with both small and large datasets.
- It supports regularization to prevent overfitting.
- It is highly customizable with many hyperparameters for tuning.

### Why Not CNN or LightGBM?
- **CNNs** are primarily used for image or sequence data, while this dataset is structured tabular data.
- **LightGBM** is another excellent option, but XGBoost was chosen for its robustness and extensive documentation.

## Outputs:
- MAE and RMSE values for model performance.
- Scatter plot of Actual vs Predicted PPV.
- Saved model file: `xgboost_model.pkl`.

## Team Members:
- **Haneen Sabra:** 202111914
- **Maha Abu Zant:** 202111934
- **Shahd Askari:** 202212836

## Libraries and Frameworks Used:
- **Libraries:** XGBoost, Pandas, Scikit-Learn, Matplotlib, Seaborn.
- **Framework:** Jupyter Notebook.

