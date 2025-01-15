import pandas as pd
import os 
import sys

#filter SNR , remove everything below 30

file = r'C:\Users\atrip\Research\Model-Interpretation-main\Data\modified_HN_P2C_20241212.csv'
df = pd.read_csv(file)
print(list(df.columns))

file_name =  os.path.split(file)
file_name = file_name[1]
#Remodeling Tobacco Labels:



# Tobacco status categories
Tob_non = {'Non-Smoker (<100 Cigarettes in Lifetime)'}
Tob_cur = {'Current Smoker', 'Currently Reformed < 6 Months (more than 7 days)'}
Tob_for = {'Reformed Smoker <= 15 Years', 'Reformed Smoker > 15 Years'}

# Initialize the new column with default values or existing column
df['Tob_label'] = 'Unknown'

# Update the 'Tob_label' based on conditions
df.loc[df['TobaccoUse'].isin(Tob_non), 'Tob_label'] = 'Never'
df.loc[df['TobaccoUse'].isin(Tob_cur), 'Tob_label'] = 'Current'
df.loc[df['TobaccoUse'].isin(Tob_for), 'Tob_label'] = 'Former'

df['Tob_label'] = df['Tob_label'].astype('category')

# Alcohol use categories
Alc_no = {'No'}
Alc_yes = {
    'Yes', 'Yes 1 std dpw', 'Yes 2 std dpw', 'Yes 3 std dpw', 'Yes 4 std dpw', 
    'Yes 5 std dpw', 'Yes 6 std dpw', 'Yes 7 std dpw', 'Yes 8 std dpw', 
    'Yes 10 std dpw', 'Yes 14 std dpw', 'Yes 21 std dpw', 'Yes 28 std dpw', 
    'Yes 37 std dpw', 'Yes 42 std dpw'
}
Alc_for = {'Not Currently'}

# Initialize the new column with default values or existing column
df['Alc_label'] = 'Unknown'

# Update the 'Alc_label' based on conditions
df.loc[df['AlcoholUse'].isin(Alc_no), 'Alc_label'] = 'No'
df.loc[df['AlcoholUse'].isin(Alc_yes), 'Alc_label'] = 'Yes'
df.loc[df['AlcoholUse'].isin(Alc_for), 'Alc_label'] = 'Not Current'

# Convert 'Alc_label' to a categorical type
df['Alc_label'] = df['Alc_label'].astype('category')


#Label categories
healthy_labels = [1, 2, 18, 20, 21, 25, 27, 32, 34, 36, 37]
cancer_labels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24, 35]
ignore_labels = [3, 4, 19, 23, 26, 28, 29, 30, 31, 33, 38, 39, 40, 41, 42]

# Initialize the new column with default values or existing column
df['ML_label'] = 'Ignore'  # Set default to 'Ignore' for those in ignore_labels or not listed

print(list(df['HighestDiagnosisLevel'].unique()))
# Update the 'ML_label' based on conditions


healthy_labels = {'Benign', 'LG Dysplasia'}
cancer_labels = {'SCC and Invasive SCC', 'HG Dysplasia'}

df.loc[df['HighestDiagnosisLevel'].isin(healthy_labels), 'ML_label'] = 'Healthy'
df.loc[df['HighestDiagnosisLevel'].isin(cancer_labels), 'ML_label'] = 'Cancer'


# df.loc[df['Diagnosis'].isin(healthy_labels), 'ML_label'] = 'Healthy'
# df.loc[df['Diagnosis'].isin(cancer_labels), 'ML_label'] = 'Cancer'



# Convert 'ML_label' to a categorical type
df['ML_label'] = df['ML_label'].astype('category')


script_dir = r'C:\Users\atrip\Research\Model-Interpretation-main\Data'
output_file_path = os.path.join(script_dir, 'ml_labelled_' + file_name)
df.to_csv(output_file_path, index=False)

print("Completed and saved \'" + str(output_file_path) + "\' to its directory.")
