import numpy as np
import pandas as pd
data = np.array([['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],
                 ['Sunny','Warm','High','Strong','Warm','Same','Yes'],
                 ['Rainy','Cold','High','Strong','Warm','Change','No'],
                 ['Sunny','Warm','High','Strong','Cool','Change','Yes']])

concepts, target = data[:,:-1], data[:,-1]
s_h = concepts[0].copy()
g_h = [["?" for _ in range(len(s_h))] for _ in range(len(s_h))]

for i, h in enumerate(concepts):
    if target[i] == "Yes":
        for x in range(len(s_h)):
            if h[x] != s_h[x]:
                s_h[x], g_h[x][x] = '?', '?'
    else:
        for x in range(len(s_h)):
            if h[x] != s_h[x]: g_h[x][x] = s_h[x]
            else: g_h[x][x] = '?'
print("Specific Boundary:", s_h)
print("General Boundary:", [g for g in g_h if g != ['?']*len(s_h)])