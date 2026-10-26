import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cols = [
    'age', 'workclass', 'fnlwgt', 'education',
    'education_num', 'marital_status', 'occupation',
    'relationship', 'race', 'sex', 'capital_gain',
    'capital_loss', 'hours_per_week', 'native_country',
    'income'
]
df = pd.read_csv('adult.csv',names=cols,header=None,skipinitialspace=True)
df.columns = df.columns.str.strip()
df = df.replace('?', pd.NA).dropna()

X = df.drop('income', axis=1)
y = df['income'].str.strip().str.rstrip('.')

num = X.select_dtypes(include='number').columns
cat = X.select_dtypes(exclude='number').columns

preprocess = ColumnTransformer([
    ('num', StandardScaler(), num),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    'Logistic Regression': make_pipeline(
        preprocess, LogisticRegression(max_iter=1000)
    ),
    'Decision Tree': make_pipeline(
        preprocess, DecisionTreeClassifier(max_depth=10, random_state=42)
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print('\n', name)
    print('Accuracy :', round(accuracy_score(y_test, pred), 3))
    print('Precision:', round(precision_score(
        y_test, pred, pos_label='>50K', zero_division=0), 3))
    print('Recall   :', round(recall_score(
        y_test, pred, pos_label='>50K', zero_division=0), 3))
    print('F1-score :', round(f1_score(
        y_test, pred, pos_label='>50K', zero_division=0), 3))

