import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.title("糖尿病分类模型")
st.write("这是一个使用 Random Forest 和贝叶斯优化的糖尿病分类模型。")

# 上传数据文件
uploaded_file = st.file_uploader("上传 CSV 数据文件", type="csv")

if uploaded_file is not None:
    # 读取数据
    data = pd.read_csv(uploaded_file)
    st.write("数据预览：", data.head())

    # 数据预处理
    features = data.columns[2:13].tolist()
    X = data[features]
    y = data['CLASS']

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 特征选择
    def mrmr_feature_selection(X, y, k=10):
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        return X_new, selected_features

    X_selected, selected_features = mrmr_feature_selection(X_scaled, y, k=10)
    st.write("选定的特征：", features)

    # 模型训练
    rf = RandomForestClassifier(random_state=42)
    search_space = {
        'n_estimators': (10, 200),
        'max_depth': (5, 30),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 10),
    }

    bayes_cv = BayesSearchCV(
        estimator=rf,
        search_spaces=search_space,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        n_iter=30,
        scoring='accuracy',
        random_state=42
    )

    bayes_cv.fit(X_selected, y)
    best_rf = bayes_cv.best_estimator_
    st.write("最佳超参数：", bayes_cv.best_params_)

    # 评估模型
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    best_rf.fit(X_train, y_train)

    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.metric("Accuracy", f"{accuracy:.4f}")
    st.metric("F1 Score", f"{f1:.4f}")

    # 绘制 ROC 曲线
    y_prob = best_rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)
