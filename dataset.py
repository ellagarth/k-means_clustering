import pandas as pd
import numpy as np

# Create a fake dataset with Student ID, GPA, Attendance Rates, and Study Hours
np.random.seed(42)  # For reproducibility

num_students = 100

student_ids = np.arange(1, num_students + 1)
gpa = np.round(np.random.uniform(2.0, 4.0, num_students), 2)  # Random GPA between 2.0 and 4.0
attendance_rate = np.round(np.random.uniform(50, 100, num_students), 2)  # Random attendance rate between 50% and 100%
study_hours = np.round(np.random.uniform(5, 40, num_students), 2)  # Random study hours between 5 and 40

# Create a DataFrame
data = {
    'Student_ID': student_ids,
    'GPA': gpa,
    'Attendance_Rate': attendance_rate,
    'Study_Hours': study_hours
}

df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('fake_student_dataset.csv', index=False)

print("CSV file 'fake_student_dataset.csv' has been created.")