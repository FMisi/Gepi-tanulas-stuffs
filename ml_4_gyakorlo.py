# -*- coding: utf-8 -*-
"""ml_4_gyakorlo.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rGWxE88NqFlRF6ZpDXSf3U8w7KqtqJEG

# Gyakorló fealdatok

1. Hajts végre egy osztályozási feladatot a [survey adatbázison](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/survey.html)
"""

import pandas as pd
df = pd.read_csv("https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/MASS/survey.csv")
df.head()

df.shape

"""ahol azt akarjuk predikálni, hogy kézkulcsolásnál melyik kéz van felül (`Fold`), az összes többi oszlop, mint jellemző alapján!"""

# Osztálycímke:
classlabel = df.Fold
# Jellemzőtér:
features = df.drop('Fold',axis=1) # a drop() eltávolít egy oszlopot (ha axis=1), létrehoz egy új dataframe-et

### Diszkrét jellemzők átkonvertálása
from sklearn import preprocessing
ohe = preprocessing.OneHotEncoder() # one hot encoding
ohe_features = ohe.fit_transform(features)

# 'Input contains NaN', valamit kezdenünk kell a NaNokkal
df.dropna().shape

# 168 példánk marad ha töröljük a NaNt tartalmazó sorokat
classlabel = df.dropna().Clap
features = df.dropna().drop('Clap',axis=1)

# így már lefut a one hot encoder
ohe_features = ohe.fit_transform(features)

### döntési fa osztályozót használunk
from sklearn import tree
dt = tree.DecisionTreeClassifier()

### Tanító- és kiértékelő adatbázisra bontás
# legyen a tanító adatbázis az első 120 elem, a többi a kiértékelő adatbázis
train_features = ohe_features[:120]
train_labels = classlabel[:120]
test_features = ohe_features[120:]
test_labels = classlabel[120:]

dt.fit(train_features, train_labels) # tanítás a tanító adatbázison
prediction = dt.predict(test_features) # predikció a kiértékelő adatbázison

### Kiértékelés találati aránnyal (accuracy)
from sklearn.metrics import accuracy_score
accuracy_score(prediction, test_labels)

### Mindig hasonlítsuk eredményeinket baseline szabályhoz!
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(train_features, train_labels)
baseline_prediction = dummy_clf.predict(test_features)
accuracy_score(baseline_prediction, test_labels)

"""2. Értékeld ki az legfeljebb 1, 2, 3 mélységűre korlátozott döntési fákat is!"""

for d in range(1,4):
  dt = tree.DecisionTreeClassifier() # fa mélységkorlátozással
  dt.fit(train_features, train_labels)
  prediction = dt.predict(test_features)
  print(d,":",accuracy_score(prediction, test_labels))