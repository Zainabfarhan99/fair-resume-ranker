import os              # To interact with the file system
import re              # For regex pattern matching (e.g., emails)
import spacy           # For NLP tasks like Named Entity Recognition
import pandas as pd    # To work with tabular data

nlp = spacy.load("en_core_web_sm")          # Load a pre-trained English NLP model from spaCy, can help with tokenization, Named entity recognition (NER), etc.

def extract_email(text):
    match = re.search(r'\S+@\S+', text)
    return match.group() if match else ""

def extract_name(text):
    lines = text.strip().split('\n')
    return lines[0] if lines else ""

def extract_skills(text):
    skills_keywords = ['python', 'java', 'sql', 'react', 'aws', 'tensorflow', 'excel', 'pandas', 'kubernetes']
    skills_found = []
    for skill in skills_keywords:
        if skill.lower() in text.lower():
            skills_found.append(skill.capitalize())
    return ", ".join(skills_found)

def extract_education(text):
    education_keywords = ['B.Tech', 'M.Tech', 'MBA', 'B.Sc', 'M.Sc']
    for line in text.split('\n'):
        for keyword in education_keywords:
            if keyword.lower() in line.lower():
                return line.strip()
    return ""

def extract_experience(text):
    for line in text.split('\n'):
        if 'experience' in line.lower() or 'year' in line.lower():
            return line.strip()
    return ""

def parse_resume(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
        return {
            "Name": extract_name(text),
            "Email": extract_email(text),
            "Skills": extract_skills(text),
            "Education": extract_education(text),
            "Experience": extract_experience(text)
        }

folder_path = "data/resumes"
parsed_data = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        resume_info = parse_resume(os.path.join(folder_path, filename))
        resume_info["Filename"] = filename
        parsed_data.append(resume_info)

df = pd.DataFrame(parsed_data)
df.to_csv("parsed_resumes.csv", index=False)
print("Parsed and saved to parsed_resumes.csv")
