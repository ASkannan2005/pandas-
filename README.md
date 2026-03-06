import pandas as pd
data = {
    "Name": ["Ram", "Sita", "Raju"],
    "Marks": [90, 80, 85]
}
df = pd.DataFrame(data)
print(df)

import pandas as pd
data = {
    "Name": ["Ram", "Sita", "Raju"],
    "Marks": [90, 80, 85]
}

Create DataFrame
df = pd.DataFrame(data)
print("Full Data:")
print(df)

Print average marks
print("\nAverage Marks:")
print(df["Marks"].mean())
 Filter students with marks > 85
print("\nStudents with Marks > 85:")
print(df[df["Marks"] > 85])

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X, y)
plt.figure(figsize=(12, 8))
plot_tree(
   clf,
  feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)
plt.show()


from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
rf = RandomForestClassifier()
rf.fit(x, y)
importances = pd.Series(rf.feature_importances_, index=iris.feature_names)
importances.nlargest(4).plot(kind='bar')
plt.title("Global Feature Importance (Random Forest)")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
iris = load_iris()
x, y = iris.data, iris.target
rf = RandomForestClassifier(random_state=42)
rf.fit(x, y)
importances = pd.Series(rf.feature_importances_, index=iris.feature_names)
importances.nlargest(4).plot(kind='bar')
plt.title("Global Feature Importance (Random Forest)")
plt.show()


import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
model = RandomForestRegressor().fit(X_train, y_train)
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.waterfall(shap_values[0])

import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
   X, y, test_size=0.2, random_state=42
)

#3clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
#explainer = shap.Explainer(clf, X_train)
#shap_values = explainer(X_test)
#shap.plots.bar(shap_values)

import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate_difference
from fairlearn.preprocessing import CorrelationRemover
from diffprivlib.models import LogisticRegression as DPLoReg
import shap
np.random.seed(42)
n_samples = 1000
gender = np.random.binomial(1, 0.5, n_samples)
income = np.random.normal(50000, 15000, n_samples)
income = income + (gender * 10000)
loan_approved = (income + np.random.normal(0, 10000, n_samples) > 55000).astype(int)
data = pd.DataFrame({'gender': gender, 'income': income, 'loan_approved': loan_approved})
X = data[['gender', 'income']]
y = data['loan_approved']
sensitive_feature = data['gender']  # 0=Female (Unprivileged), 1=Male (Privileged)
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(solver='liblinear')
lr.fit(X_train_scaled, y_train)
preds = lr.predict(X_test_scaled)
print("--- Initial Model (Before Fairness & Privacy) ---")
print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
# Selection Rate Difference: Should be close to 0 for fairness
sr_diff = selection_rate_difference(y_test, preds, sensitive_features=s_test)
print(f"Selection Rate Difference (Bias): {sr_diff:.2f}")
cr = CorrelationRemover(sensitive_feature_ids=['gender'])
X_train_fair = cr.fit_transform(X_train)
X_test_fair = cr.transform(X_test)
dp_lr = DPLoReg(epsilon=1.0, data_norm=10)
dp_lr.fit(X_train_fair, y_train)
dp_preds = dp_lr.predict(X_test_fair)

print("\n--- Model After Fairness Mitigation & Differential Privacy ---")
print(f"DP Accuracy: {accuracy_score(y_test, dp_preds):.2f}")
dp_sr_diff = selection_rate_difference(y_test, dp_preds, sensitive_features=s_test)
print(f"New Selection Rate Difference: {dp_sr_diff:.2f}")
print("\n--- Explainability (SHAP values) ---")
explainer = shap.LinearExplainer(dp_lr, X_train_fair)
shap_values = explainer.shap_values(X_test_fair)
