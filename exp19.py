import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Salary, CreditScore -> LoanApproved(0,1)
df = pd.DataFrame({'Sal': [40, 70, 20, 90], 'Cred': [600, 750, 500, 800], 'Loan': [0, 1, 0, 1]})
model = GaussianNB().fit(df[['Sal', 'Cred']], df['Loan'])
print("Loan Approval Prediction:", "Approved" if model.predict([[65, 700]])[0] == 1 else "Rejected")