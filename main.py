import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
df = pd.read_csv('StudentsPerformance.csv')

# Convert categorical columns to numeric
encoder = LabelEncoder()
df['enc_gender'] = encoder.fit_transform(df['gender'])
df['enc_lunch'] = encoder.fit_transform(df['lunch'])
df['enc_test_preparation_course'] = encoder.fit_transform(df['test preparation course'])

# Create dummy variables for categorical columns
dum_df = pd.get_dummies(df[['race/ethnicity', 'parental level of education']])
df = pd.concat([df, dum_df], axis=1)

# Compute average score from math, reading, and writing scores
df['score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)
