import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skopt import BayesSearchCV
import matplotlib.pyplot as plt


# Load the data导入数据
data = pd.read_csv("D:\\HuaweiMoveData\\Users\\86178\\Desktop\\Dataset of Diabetes2.csv")  # Replace with your dataset path
data


# 提取需要的特征列，X是除了ID、No_Pation以外的11个指标
features =data.columns[2:13].tolist()  
X = data[features]
y = data['CLASS']


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Step 2: Feature Selection with MRMR
def mrmr_feature_selection(X, y, k=10):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_new, selected_features

X_selected, selected_features = mrmr_feature_selection(X_scaled, y, k=10)
print("Selected features:", X.columns[selected_features])


# Define the random forest classifier
rf = RandomForestClassifier(random_state=42)

# Define the hyperparameter search space
search_space = {
    'n_estimators': (10, 200),
    'max_depth': (5, 30),
    'min_samples_split': (2, 10),
    'min_samples_leaf': (1, 10)
}

# Define BayesSearchCV
bayes_cv = BayesSearchCV(
    estimator=rf,
    search_spaces=search_space,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_iter=30,  # Number of parameter combinations to test
    scoring='accuracy',
    random_state=42
)

# Perform hyperparameter tuning
bayes_cv.fit(X_selected, y)
best_rf = bayes_cv.best_estimator_
print("Best Parameters:", bayes_cv.best_params_)


# Step 4: Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
best_rf.fit(X_train, y_train)


# Step 5: Model Evaluation
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), multi_class="ovr")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# ROC Curve
fpr = {}
tpr = {}
for i in range(3):  # Three classes
    fpr[i], tpr[i], _ = roc_curve(pd.get_dummies(y_test).iloc[:, i], pd.get_dummies(y_pred).iloc[:, i])

plt.figure()
for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Class {i}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()