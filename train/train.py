import os
import pathlib
import pandas as pd
import pickle as pkl

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Get the project dir
project_dir = pathlib.Path().resolve().parent

# Get the csv file containing information about the maladies cardiaques
maladie_cardiaque_csv = os.path.join(project_dir, "sujet", "maladie_cardiaque.csv")

# Read the dataframe of maladie_cardiaque.csv
df = pd.read_csv(maladie_cardiaque_csv)

# Drop the Unamed 0 and ID columns
df = df.drop(columns=['Unnamed: 0'])
df = df.drop(columns=["id"])

# Create slice for the age column
df["age"] = round(df["age"] / 365)  # Transform the age column from days to year
ages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
df['age'] = pd.cut(df['age'], bins=ages, labels=False)

# Create slice for the height column
tailles = [0, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250]
df['taille'] = pd.cut(df['taille'], bins=tailles, labels=False)

# Create slice for the weight column
poids = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
df['poids'] = pd.cut(df['poids'], bins=poids, labels=False)

# Instantiates a label encoder that will be used for some columns
le = LabelEncoder()

# Encode the cholesterol column which is a string
df['cholesterol'] = le.fit_transform(df['cholesterol'])

# Check if we have columns that contain NaN values
# cols_with_na = df.columns[df.isna().any()].tolist()
# print(f"Columns with NaN values => {cols_with_na}")

# Verificate if the target value is binary
# print(f"Target variable distinct values => {len(df['malade'].unique())}")

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('malade', axis=1),
    df['malade'],
    test_size=0.2
)

# Instantiates a linear regression model
cls_list = [
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

# Selection the best classifier
res_cls = {}
res_accuracy = {}
for model in cls_list:
    # Train the model with our split data
    model.fit(x_train, y_train)

    # Get some prediction on the data
    y_pred = model.predict(x_test)

    # Add to res
    res_accuracy[model.__class__.__name__] = accuracy_score(y_test, y_pred)
    res_cls[model.__class__.__name__] = model

print(f"The best accuracy score is obtained by the classifier : {max(res_accuracy)}")
for k, v in res_accuracy.items():
    print(f"{k} => {v}")

# Save the model in the app directory, to get used
with open(os.path.join(project_dir, "app", "models", "label_encoder.pkl"), "wb") as f:
    print("The label encoder has been saved")
    pkl.dump(le, f)

# Save the model in the app directory, to get used
with open(os.path.join(project_dir, "app", "models", "model.pkl"), "wb") as f:
    print("The trained model has been saved")
    pkl.dump(res_cls[max(res_accuracy)], f)
