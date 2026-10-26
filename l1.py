
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DPF', 'Age', 'Outcome']
df = pd.read_csv('pima_diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

m1 = LogisticRegression(max_iter=1000)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)

sc = StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s = sc.transform(X_test)

m2 = LogisticRegression(max_iter=1000)
m2.fit(X_train_s, y_train)
p2 = m2.predict(X_test_s)

for name, p in [('Without Scaling', p1), ('With Scaling', p2)]:
    print('\n', name)
    print('Accuracy :', accuracy_score(y_test, p))
    print('Precision:', precision_score(y_test, p, zero_division=0))
    print('Recall   :', recall_score(y_test, p, zero_division=0))
    print('F1-score :', f1_score(y_test, p, zero_division=0))
