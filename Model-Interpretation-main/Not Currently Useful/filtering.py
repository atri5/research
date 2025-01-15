import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

import timeit

t0 = timeit.default_timer()
# Load data
T = pd.read_csv('HN_P2C_20240506.csv')
#print(T.info())
t1 = timeit.default_timer()
print("Opened file in: "+ str(t1 - t0) + " seconds.")


idx_cancer = T[T['ML_label'] == 'Cancer'].index
idx_healthy = T[T['ML_label'] == 'Healthy'].index
idx_lymphoid = T[T['ML_label'] == 'Lymphoid'].index

# Select subsets regarding Tissue Classification: Cancer, Healthy, Lymphoid
T_all = T
T_cancer = T.loc[idx_cancer]
T_healthy = T.loc[idx_healthy]
T_lymphoid = T.loc[idx_lymphoid]


# Separate Feature and Target Variables

X_all = T.iloc[:, [1, 4, 7, 16, 17, 18, 22, 23, 24, 28] + list(range(28, 64))]
X_cancer = T_cancer.iloc[:, [1, 4, 7, 16, 17, 18, 22, 23, 24, 28] + list(range(28, 64))]
X_healthy = T_healthy.iloc[:, [1, 4, 7, 16, 17, 18, 22, 23, 24, 28] + list(range(28, 64))]
X_lymphoid = T_lymphoid.iloc[:, [1, 4, 7, 16, 17, 18, 22, 23, 24, 28] + list(range(28, 64))]


#to investigate subgroups of features/predictor
var_LT = { 'lifet_avg_ch1','lifet_avg_ch2', 'lifet_avg_ch2' }
var_IR = {'int_ratio_ch1',  'int_ratio_ch2',  'int_ratio_ch2'   }


# Target variable for Cancer subset
Y_p16 = T_cancer['HPVstatus']

#Assign Features
X = X_cancer
Y = Y_p16

# Splitting into training and testing sets, considering patient groups
# Separate subjects into two groups based on some condition (e.g., Case)
subjectsGroup1 = T_cancer[T_cancer['Case'] < 101]
subjectsGroup2 = T_cancer[T_cancer['Case'] >= 101]

numSubjectsGroup1 = len(subjectsGroup1['Case'].unique())
numSubjectsGroup2 = len(subjectsGroup2['Case'].unique())


trainRatio = 0.8 #as specified, tune to your liking
numTrainingSubjects1 = int(np.round(trainRatio * numSubjectsGroup1))
numTrainingSubjects2 = int(np.round(trainRatio * numSubjectsGroup2))

# Randomize subjects by sampling without replacement
randSubjectsGroup1 = subjectsGroup1['Case'].sample(n=numTrainingSubjects1, random_state=1).index
randSubjectsGroup2 = subjectsGroup2['Case'].sample(n=numTrainingSubjects2, random_state=1).index

trainIdx1 = subjectsGroup1['Case'].isin(subjectsGroup1.loc[randSubjectsGroup1[:numTrainingSubjects1], 'Case'])
trainIdx2 = subjectsGroup2['Case'].isin(subjectsGroup2.loc[randSubjectsGroup2[:numTrainingSubjects2], 'Case'])

# Convert boolean masks to numpy arrays for concatenation
trainIdx1 = trainIdx1.values
trainIdx2 = trainIdx2.values

trainIdx = np.concatenate([trainIdx1, trainIdx2])

# Randomly assign training and testing sets
train_idx1 = np.random.rand(len(subjectsGroup1)) < 0.8
train_idx2 = np.random.rand(len(subjectsGroup2)) < 0.8


X_train = X[trainIdx]
Y_train = Y[trainIdx]

X_test = X[~trainIdx]
Y_test = Y[~trainIdx]


# Initialize and fit AdaBoost with Decision Trees
t0 = timeit.default_timer()
np.random.seed(1)
tree = DecisionTreeClassifier(random_state = 1)  # Ensuring reproducibility
model = AdaBoostClassifier(
    base_estimator=tree,
    n_estimators=100,  # This corresponds to 'NumLearningCycles' in MATLAB
    learning_rate=0.01,
    random_state=42
)

model.fit(X_test[:, 7:49], Y_test) 
t1 = timeit.default_timer()

print("Fit model in: "+str(t1-t0) + " seconds.")


# View the first trained model
first_tree = model.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(first_tree, filled=True)
plt.show()

# Predict labels and scores
labels, scores = model.predict(X_test.iloc[:, 7:49]), model.predict_proba(X_test.iloc[:, 7:49])

# Extract true class labels from a pandas DataFrame assuming binary classification
true_labels = X_test['ML_label']

# Calculate ROC curve and AUC for the first class
fpr, tpr, thresholds = roc_curve(true_labels, scores[:, 1], pos_label=model.classes_[0])
roc_auc = auc(fpr, tpr)

# Create a DataFrame for ROC data similar to rocmetrics
roc_data = pd.DataFrame({
    'FPR': fpr,
    'TPR': tpr,
    'Thresholds': thresholds,
    'ClassName': [model.classes_[0]] * len(fpr)
})
class_to_view = model.classes_[0]  # Change index to [1] for second class, etc(equivalent to idx in original)
roc_metrics_df = pd.DataFrame({
    'FPR': roc_data[class_to_view]['fpr'],
    'TPR': roc_data[class_to_view]['tpr'],
    'Thresholds': roc_data[class_to_view]['thresholds']
})
print(roc_metrics_df.head())
plt.figure()
plt.plot(roc_data[class_to_view]['fpr'], roc_data[class_to_view]['tpr'], label=f'ROC curve for {class_to_view} (area = {roc_data[class_to_view]["AUC"]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()




# Placeholder for SHAP values analysis
# Currently, Python does not have a direct equivalent of MATLAB's 'shapley' function.
# You can use the SHAP library in Python for equivalent functionality.

explainer = shap.TreeExplainer(model, data=X_test)
# Compute SHAP values for the entire test set
shap_values = explainer.shap_values(X_test)

# Plot summary of SHAP values for all features across all data points
shap.summary_plot(shap_values, X_test)

# Compute SHAP values for a specific instance, e.g., the first instance
specific_shap_values = explainer.shap_values(X_test.iloc[0, :])

# Plot the SHAP values for this specific instance
shap.force_plot(explainer.expected_value, specific_shap_values, X_test.iloc[1, :], matplotlib=True)

# Print the SHAP values for the specific instance
print("SHAP values for the specific instance:")
print(specific_shap_values)
