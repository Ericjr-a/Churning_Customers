# Churning_Customers

EricAfari_Assignment3.ipynb file:
My project started with data preprocessing, where the missing values in a column called Total Charges are handled by converting them to numeric values and dropping rows with Nan values.
The column called customer ID is also dropped because it will not be as useful when predicting. 
Next is feature engineering, where label encoding converts the categorical values to numerical ones. With this, the target variable churn is also encoded. 
A correlation matrix is used to identify the relationship between different features. Also, other features are plotted with the target variable churn using count plots and box plots. The features that strongly correlate with the target variable will be chosen for the churn prediction app.
After eda, the model is built, and the features are scaled using a Standard scaler. I built the model using Keras to create a deep learning model. After GridSearchCV is used for hyperparameter tuning. From the GridSearchCV, I select the model that has the best score.
After this, the model’s performance is evaluated on the tested data. The accuracy and AUC score of the model are calculated. Finally, the best model is saved for deployment.

Streamlitapp.py file:
This creates an app for customer prediction where the keras and scaler models are loaded. Then, the chosen and scaled features are used here to predict the churn rate. The categorical inputs are converted into numerical inputs based on their index in the list> When the predict button is used , the selected features are rescaled, encoded, and scaled. The churn probability is then displayed with the tested model’s confidence level and accuracy score.
