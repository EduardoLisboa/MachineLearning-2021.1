import pandas as pd
from impyute import mice

df = pd.read_csv('diabetes_dataset.csv')

rmv_missings = df[df['BloodPressure'].notnull()]
rmv_missings = rmv_missings[rmv_missings['BMI'].notnull()]
rmv_missings = rmv_missings[rmv_missings['Glucose'].notnull()]
rmv_outliers = rmv_missings[rmv_missings['Pregnancies'] < 13]
rmv_outliers = rmv_outliers[rmv_outliers['Age'] < 67]

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'BMI', 'DiabetesPedigreeFunction', 'Age']

X_imputed = mice(rmv_outliers[feature_cols].values)

X = pd.DataFrame(X_imputed, columns=feature_cols)

X.to_csv("X_train.csv")

rmv_outliers['Outcome'].to_csv("Y_train.csv")