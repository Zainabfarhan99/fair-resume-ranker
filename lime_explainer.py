# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import train_test_split
# from lime.lime_text import LimeTextExplainer

# # Load CSV
# df = pd.read_csv("ranked_resumes.csv")

# # Combine resume fields into a single text column
# df['resume_text'] = (
#     df['Skills'].fillna('') + ' ' +
#     df['Education'].fillna('') + ' ' +
#     df['Experience'].fillna('')
# )

# # Binary classification: Top 50% score = selected
# df['label'] = (df['Score'] >= df['Score'].median()).astype(int)

# texts = df['resume_text'].values
# labels = df['label'].values

# # Split
# X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# # Build pipeline: TF-IDF + Classifier
# pipeline = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
# pipeline.fit(X_train, y_train)

# # LIME explainer
# class_names = ['Not Selected', 'Selected']
# explainer = LimeTextExplainer(class_names=class_names)

# # Explain one test instance
# idx = 1
# exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba, num_features=10)
# exp.save_to_file('lime_explanation.html')

# print("Explanation saved to lime_explanation.html")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_text import LimeTextExplainer
import os
import matplotlib.pyplot as plt

# Load the resume data
df = pd.read_csv("ranked_resumes.csv")

# Combine text fields to simulate "resume text"
df["resume_text"] = df["Skills"].fillna('') + " " + df["Education"].fillna('') + " " + df["Experience"].fillna('')

texts = df["resume_text"]
labels = df["Score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a model
model = RandomForestClassifier()
model.fit(X_train_vec, y_train)

# Create LimeTextExplainer
explainer = LimeTextExplainer(class_names=["Low", "High"])

# Create output folder
os.makedirs("lime_outputs", exist_ok=True)

# Define a wrapper for the model
def predict_proba_wrapper(texts):
    return model.predict_proba(vectorizer.transform(texts))

# Now generate LIME explanations
for i in range(len(X_test)):
    print(f"Explaining resume #{i}")
    exp = explainer.explain_instance(
        X_test.iloc[i],
        predict_proba_wrapper,  # use wrapper here
        num_features=10
    )
    exp.save_to_file(f"lime_outputs/lime_explanation_{i}.html")

print("LIME explanations saved in lime_outputs/")

