from google.colab import files
data_to_load=files.upload()

import pandas as pd
import plotly.express as px
import plotly.graph_objects as gp
import csv
import plotly.figure_factory as pf
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sb

df = pd.read_csv("BankNote_Authentication.csv")
print(df.head())

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

scores = df[["variance","skewness","curtosis","entropy"]]
print(scores)
results = df["class"]
score_train, score_test, results_train, results_test = tts(scores, results, test_size= 0.25, random_state = 6)

model = LogisticRegression(random_state = 0)
model.fit(score_train, results_train)
pred = model.predict(score_test)
predict_value_1 = []
for i in pred:
    if i == 0:
        predict_value_1.append("Authorized")
    else:
        predict_value_1.append("Forged")
actual_value_1 = []
for i in results_test.ravel():
    if i == 0:
        actual_value_1.append("Authorized")
    else:
         actual_value_1.append("Forged")
labels = ["Authorized", "Forged"]
cm =confusion_matrix(actual_value_1, predict_value_1, labels)
ax = plt.subplot()
sb.heatmap(cm, annot = True, ax = ax)

tn, fp, fn, tp = confusion_matrix(results_test, pred).ravel()
print("True Negatives: ",tn)
print("True Positives: ",tp)
print("False Positives: ",fp)
print("False Negatives: ",fn)

Accuracy = (tn+tp)*100/(tp+tn+fp+fn)
print("Accuracy :", (Accuracy))
