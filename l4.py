import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel('Online Retail.xlsx')

df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['Amount'] = df['Quantity'] * df['UnitPrice']

customer = df.groupby('CustomerID').agg(
    TotalAmount=('Amount', 'sum'),
    TotalQuantity=('Quantity', 'sum'),
    Purchases=('InvoiceNo', 'nunique')
).reset_index()

customer['Segment'] = pd.qcut(
    customer['TotalAmount'], q=3,
    labels=['Low', 'Medium', 'High'],
    duplicates='drop'
)

X = customer[['TotalQuantity', 'Purchases']]
y = LabelEncoder().fit_transform(customer['Segment'])

model = DecisionTreeClassifier(
    criterion='entropy', max_depth=3, random_state=42
)
model.fit(X, y)

plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=X.columns,
          class_names=['Low', 'Medium', 'High'],
          filled=True, rounded=True)
plt.show()

for name, importance in zip(X.columns, model.feature_importances_):
    print(name, ":", round(importance, 3))