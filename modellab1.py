import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

# --- Data Preparation ---
data = {'Income': ['High', 'Medium', 'Low', 'Medium'],
        'Debt': ['Low', 'Medium', 'High', 'Low'],
        'PaymentHistory': ['Good', 'Average', 'Poor', 'Good'],
        'Score': ['High', 'Medium', 'Low', 'High']}
df = pd.DataFrame(data)
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
X, y = df_encoded.drop('Score', axis=1), df_encoded['Score']


model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)


new_data = [[1, 2, 0]] 
prediction = model.predict(new_data)
print(f"ID3 Decision Tree Prediction: {le.inverse_transform(prediction)[0]}")
print("Tree Logic:\n", export_text(model, feature_names=['Income', 'Debt', 'Payment']))