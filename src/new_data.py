import pandas as pd
import os

# Read original cleaned dataset
df = pd.read_csv("data/preprocessed_data/cleaned_student_dataset.csv")

# Take 10% random sample as "new incoming data"
new_data = df.sample(frac=0.1, random_state=42)

# Ensure the folder exists
os.makedirs("data/new_data", exist_ok=True)

# Save new simulated data
new_data.to_csv("data/new_data/new_student_data.csv", index=False)

print("✅ New incoming data saved to data/new_data/new_student_data.csv")
