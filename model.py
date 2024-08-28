import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('medical_insurance.csv')

# Label Encoding for categorical features
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])

le_smoker = LabelEncoder()
df['smoker'] = le_smoker.fit_transform(df['smoker'])

# One-Hot Encoding for 'region'
df = pd.get_dummies(df, columns=['region'])

# Features and target
X = df.drop('charges', axis=1).values
y = df['charges'].values

# Feature Scaling
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

sc_y = StandardScaler()
y = y.reshape(-1, 1)  # Reshape y for scaling
y = sc_y.fit_transform(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Polynomial Regression
poly = PolynomialFeatures(degree=4)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
lr_poly = LinearRegression()
lr_poly.fit(X_poly_train, y_train)

# Support Vector Regression
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train.ravel())  # Flatten y_train to fit SVR

# Decision Tree Regression
dt = DecisionTreeRegressor(random_state=0)
dt.fit(X_train, y_train)

# Random Forest Regression
rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(X_train, y_train.ravel())

# Save each model and preprocessor into separate files
models = {
    'linear_regression.pkl': lr,
    'polynomial_features.pkl': poly,
    'svr.pkl': svr,
    'decision_tree.pkl': dt,
    'random_forest.pkl': rf,
    'label_encoder_sex.pkl': le_sex,
    'label_encoder_smoker.pkl': le_smoker,
    'scaler_X.pkl': sc_X,
    'scaler_y.pkl': sc_y
}

for filename, obj in models.items():
    with open(f'model/{filename}', 'wb') as f:
        pickle.dump(obj, f)

print("All models and preprocessors have been saved in separate files.")
