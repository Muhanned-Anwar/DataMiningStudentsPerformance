import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load and prepare data
df = pd.read_csv('StudentsPerformance.csv')

# Convert categorical columns to numeric
encoder = LabelEncoder()
df['enc_gender'] = encoder.fit_transform(df['gender'])
df['enc_lunch'] = encoder.fit_transform(df['lunch'])
df['enc_test_preparation_course'] = encoder.fit_transform(df['test preparation course'])
