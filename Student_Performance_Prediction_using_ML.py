import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv(
    r"C:\Users\risha\Downloads\student-mat-clean (1)RT.csv"
)

print("Dataset Shape:", dataset.shape)

print("\nFirst 5 Rows:")
print(dataset.head())

print("\nColumn Names:")
print(dataset.columns.tolist())

print("\nTotal Missing Values:")
print(dataset.isnull().sum().sum())

# Step 3: Create Pass/Fail Target

dataset["Pass_Fail"] = dataset["G3"].apply(
    lambda grade: "Pass" if grade >= 10 else "Fail"
)

print("\nPass/Fail Distribution:")
print(dataset["Pass_Fail"].value_counts())

print("\nPass/Fail Percentage:")
print(dataset["Pass_Fail"].value_counts(normalize=True) * 100)



# Step 4: Data Visualization

# Graph 1: Pass/Fail Count
plt.figure(figsize=(6, 4))

sns.countplot(
    data=dataset,
    x="Pass_Fail",
    hue="Pass_Fail",
    legend=False
)

plt.title("Student Pass/Fail Distribution")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# Graph 2: Final Grade Distribution
plt.figure(figsize=(7, 4))

sns.histplot(
    data=dataset,
    x="G3",
    bins=11,
    kde=True
)

plt.title("Distribution of Final Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()


# Graph 3: Study Time vs Final Grade
plt.figure(figsize=(7, 4))

sns.boxplot(
    data=dataset,
    x="studytime",
    y="G3"
)

plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()
plt.show()





# Step 5: Prepare Features and Target

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Remove grade columns to avoid data leakage
X = dataset.drop(columns=["G1", "G2", "G3", "Pass_Fail"])

# Target variable
y = dataset["Pass_Fail"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())




# Step 6: Train-Test Split and Preprocessing

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object","str"]
).columns

# Numerical data preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Display results
print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)





# Step 7: Train Five Machine Learning Algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

# Dictionary to store all five models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        random_state=42
    )
}

# Dictionary to store trained pipelines
trained_models = {}

# Train each model
for name, model in models.items():

    # Combine preprocessing and machine learning model
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train the model using training data
    pipeline.fit(X_train, y_train)

    # Store the trained pipeline
    trained_models[name] = pipeline

    print(name, "trained successfully!")

print("\nAll five machine learning models have been trained.")



# Step 8: Evaluate All Five Machine Learning Models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Store evaluation results
results = []

# Evaluate each trained model
for name, model in trained_models.items():

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, pos_label="Pass", zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, pos_label="Pass", zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, pos_label="Pass", zero_division=0
    )

    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "Precision": round(precision * 100, 2),
        "Recall": round(recall * 100, 2),
        "F1 Score": round(f1 * 100, 2)
    })

    print("\n" + "=" * 50)
    print("Algorithm:", name)
    print("=" * 50)

    print("\nConfusion Matrix:")
    print(confusion_matrix(
        y_test, y_pred, labels=["Fail", "Pass"]
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, zero_division=0
    ))

# Display comparison table
import pandas as pd

results_df = pd.DataFrame(results)

print("\n\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))



# Step 9: Visualize Model Performance

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -----------------------------------
# 1. Accuracy Comparison Chart
# -----------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Algorithm"],
    results_df["Accuracy"]
)

plt.title("Accuracy Comparison of Five ML Algorithms")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


# -----------------------------------
# 2. Precision, Recall and F1 Score
# -----------------------------------

results_df.plot(
    x="Algorithm",
    y=["Precision", "Recall", "F1 Score"],
    kind="bar",
    figsize=(12, 6)
)

plt.title("Precision, Recall and F1 Score Comparison")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
plt.xticks(rotation=20)
plt.legend(loc="best")
plt.tight_layout()
plt.show()


# -----------------------------------
# 3. SVM Confusion Matrix Heatmap
# -----------------------------------

best_model = trained_models["SVM"]

y_pred_svm = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred_svm,
    labels=["Fail", "Pass"]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fail", "Pass"],
    yticklabels=["Fail", "Pass"]
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

#print("Step 9 completed successfully!")






# Step 10: Actual vs Predicted Results

import pandas as pd

# Create comparison DataFrame
prediction_results = pd.DataFrame({
    "Actual Result": y_test,
    "Predicted Result": y_pred_svm
})

# Display first 20 predictions
print("Actual vs Predicted Results:")
print(prediction_results.head(20).to_string(index=False))

# Check correct and incorrect predictions
correct_predictions = (
    prediction_results["Actual Result"] ==
    prediction_results["Predicted Result"]
).sum()

total_predictions = len(prediction_results)

print("\nTotal Predictions:", total_predictions)
print("Correct Predictions:", correct_predictions)
print("Incorrect Predictions:", total_predictions - correct_predictions)

print("\nStep 10 completed successfully!")












