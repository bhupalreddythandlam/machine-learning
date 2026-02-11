import csv
# Load data (assuming data.csv: Col1-N are features, Last Col is Label)
data = [['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
        ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
        ['Rainy','Cold','High','Strong','Warm','Change','No'],
        ['Sunny','Warm','High','Strong','Cool','Change','Yes']]

hypothesis = ['0'] * 6 # Initialize with most specific
for row in data:
    if row[-1] == "Yes":
        if hypothesis == ['0'] * 6: hypothesis = row[:-1].copy()
        for i in range(len(hypothesis)):
            if row[i] != hypothesis[i]: hypothesis[i] = '?'
print("Final Hypothesis:", hypothesis)