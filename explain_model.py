import shap
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load CSV
df = pd.read_csv('ranked_resumes.csv')
print(df.columns)

# Drop name column if present
if 'Name' in df.columns:
    df = df.drop(columns=['Name'])

# Combine textual features
text_columns = ['Skills', 'Experience', 'Education']  # Adjust as per actual CSV
X_text = df[text_columns].fillna('').agg(' '.join, axis=1)

# Target
y = df['Score']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=300)
X_vect = vectorizer.fit_transform(X_text).toarray()

# Scale target
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).ravel()

# Train model
model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X_vect, y_scaled, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# Plot
shap.summary_plot(shap_values, X_test, feature_names=vectorizer.get_feature_names_out())
