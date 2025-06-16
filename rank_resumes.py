import pandas as pd

# Load parsed resumes
df = pd.read_csv('parsed_resumes.csv')
print(df.columns)
# Load job description
with open('job_description.txt', 'r') as f:
    jd = f.read().lower()

# Simple scoring: count matching keywords in 'Skills'
def score(skills):
    return sum(1 for word in jd.split() if word in str(skills).lower())

df['Score'] = df['Skills'].apply(score)

# Sort by score (descending)
df = df.sort_values(by='Score', ascending=False)

# Save the ranked results
df.to_csv('ranked_resumes.csv', index=False)

print("Ranking complete. Saved to ranked_resumes.csv")
