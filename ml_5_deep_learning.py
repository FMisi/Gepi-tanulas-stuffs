# -*- coding: utf-8 -*-
"""ml_5_deep_learning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1eI8j0CRj2L8fbDbat5Dl35FC0Bg9BU5u

# Deep Learning
Olvasd el az [elméleti bevezetőt](http://inf.u-szeged.hu/~rfarkas/deep_learning.html).

### Futtatás GPU-n

A mély neurális hálók tanítása nagyon számításigényes, viszont visszavezetve mátrixműveletekre nagyon jól párhuzamosítható GPU-n. Érdemes a Google Colab-ban is átváltani GPU-ra. Ezt az Edit>Notebook settings menüben tehetjük meg GPU-t választva hardveres gyorsításra. Ha CPU-ról átvátunk GPU-ra akkor újra kell futtatni a teljes notebookot!

A Cuda egy alacsony szintű szoftverréteg mátrixműveletek GPU-n való nagyon hatékony megvalósítására. E fölé épülnek a deep learning keretrendszerek, pl.  [PyTorch](https://pytorch.org/) és a [Tensorflow](https://www.tensorflow.org/).
"""

### PyTorch deep learning keretrendszert használjuk: https://pytorch.org
import torch

### Futtatási környezet előkészítése

# Cuda inicializálása
torch.backends.cudnn.deterministic = True

# a neurális hálók tanításánál a véletlenszám-generálásnak nagy szerepe van
# érdemes a random seedet fixálni, hogy minden futtatásra ugyanazt az eredményt kapjuk
SEED = 202004
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

"""# Szövegosztályozás mély tanulással

Az [elöző](https://colab.research.google.com/drive/1Ve2FOeA7ceEgS0eqL-31CUFFT33s_PDm) órán megoldott szövegosztályozási feladatra fogunk adni itt egy mély gépi tanulási megoldást. Ugyanaz a feladat, véleményosztályozás. Ugyanazon az adatbázison, ugyanazon kiértékelési metrikát használjuk, így az eredmények összehasonlíthatóak a klasszikus gépi tanulási eredményekkel.
"""

import pandas as pd
train_data = pd.read_csv('https://github.com/rfarkas/student_data/raw/main/sentiment/train.tsv', sep='\t')
test_data  = pd.read_csv('https://github.com/rfarkas/student_data/raw/main/sentiment/test.tsv', sep='\t')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer = CountVectorizer()
cv_counts = vectorizer.fit_transform(train_data.text)
idf_transformer = TfidfTransformer(use_idf=True).fit(cv_counts)
features = idf_transformer.transform(cv_counts)
test_features = idf_transformer.transform(vectorizer.transform(test_data.text))

features

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
model = SGDClassifier().fit(features, train_data.label)
accuracy_score(y_true=test_data.label, y_pred=model.predict(test_features))

"""## Egyszerű neurális hálózat"""

### ritka mátrixot tensor formátumra alakítjuk
import numpy as np
X_train_tensor = torch.from_numpy(features.todense()).float()
X_test_tensor  = torch.from_numpy(test_features.todense()).float()

train_data.label

### PyTorch-ban még a célváltozó sem lehet diszkrét...
### A LabelEncoder véletlenszerűen Int-eket rendel az egyes értékekhez
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y_train_tensor = torch.as_tensor(le.fit_transform(train_data.label))
Y_test_tensor  = torch.as_tensor(le.transform(test_data.label))

### Jellemzőtér (=bemeneti réteg) dimenziói és célváltozók száma (=kimeneti réteg dimenziója)
VOCAB_SIZE = len(vectorizer.vocabulary_)
OUT_CLASSES = 3

### Linear Machine, LM
### A legegyszerűbb neurális háló (ami megegyezik a lineáris géppel)
### a kimeneti neuron össze vannak kötve a bementiekkel (mindegyik mindegyikkel)

class LM_Network(torch.nn.Module):
     def __init__(self,vocab_size,out_classes):
        super().__init__()
        self.linear = torch.nn.Linear(vocab_size,out_classes)
     def forward(self,x):
        return self.linear(x)

model = LM_Network(VOCAB_SIZE,OUT_CLASSES)
print(model)

# összesen ennyi paramétert kell tanítanunk:
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)

#predikció a random hálóval
model(X_train_tensor[1:3])

### Multi Layer Perceptron, MLP
### 1 rejtett réteget tartalmazó neuárlis hálózat

class MLP_Network(torch.nn.Module):
  def __init__(self,vocab_size,hidden_units,num_classes):
      super().__init__()
      #First fully connected layer
      self.fc1 = torch.nn.Linear(vocab_size,hidden_units)
      #Second fully connected layer
      self.fc2 = torch.nn.Linear(hidden_units,num_classes)
      #Final output of sigmoid function
      self.sigmoid = torch.nn.Sigmoid()

  def forward(self,x):
      y1 = self.sigmoid(self.fc1(x))
      output = self.sigmoid(self.fc2(y1))
      return output

HIDDEN_UNITS = 100
model = MLP_Network(VOCAB_SIZE, HIDDEN_UNITS, OUT_CLASSES)
print(model)
print(count_parameters(model), "tanulandó paraméter")

### Kiértékelő függvény
def accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True) # a 3 osztályra adott kimeneti érték közül melyik a legnagyobb
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum(dtype=float) / y.shape[0]

### ha az epoch végén egy független validációs halmazon is ki akarjuk értékelni a modellt:
def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()  # inicializálás
    with torch.no_grad():
        for batch in iterator:
            # predikció
            predictions = model(batch[0])
            # kiértékelés
            acc = accuracy(predictions, batch[1].long())
            epoch_acc += acc.item()

    return epoch_acc / len(iterator)

Y_test_tensor

from torch.utils.data import Dataset, TensorDataset
train_data = TensorDataset(X_train_tensor, Y_train_tensor)
test_data  = TensorDataset(X_test_tensor,  Y_test_tensor)

### Ha egy adatbázison akarunk végigmenni akkor ahhoz iterátort kell definiálni
from torch.utils.data import DataLoader
train_loader = DataLoader(train_data,batch_size=16, shuffle=True)

# random hálózat kiértékelése az egész adatbázison
evaluate(model, train_loader)

### A tanítás során többször végigmegyünk a tanító adatbázison (egy kör egy epoch)
def train(model, iterator, optimizer, criterion):
    # minden epoch végén ellenőrízni fogjuk az accuracyt
    epoch_acc = 0

    model.train() # inicializálás
    for batch in iterator:
        # predikáljuk le a tanító példákat az aktuális paraméterekkel:
        optimizer.zero_grad()
        predictions = model(batch[0])

        # a háló aktuális paraméterivel ennyi a hiba a batchen:
        loss = criterion(predictions, batch[1].long())
        acc = accuracy(predictions, batch[1].long())

        # hibavisszaterjesztéssel (backpropagation) javítunk a paramétereken:
        loss.backward()
        optimizer.step()

        epoch_acc += acc.item()

    return epoch_acc / len(iterator)

# Commented out IPython magic to ensure Python compatibility.
# ### Neurális hálózat tanítása
# %%time
# NUM_EPOCHS = 10
# BATCH_SIZE = 64
# 
# #Neurális háló architektúra megadása
# model = MLP_Network(VOCAB_SIZE,HIDDEN_UNITS,OUT_CLASSES)
# 
# #optimalizáló eljárás
# import torch.optim as optim
# optimizer = optim.Adam(model.parameters()) # ADAM optimalizáló algoritmus
# 
# #célfüggvény
# import torch.nn as nn
# loss_fun = nn.CrossEntropyLoss()
# 
# iterator = DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)
# for i in range(NUM_EPOCHS):
#    print(i, ". epoch acc:", train(model, iterator, optimizer, loss_fun))

### Kiértékelés a teszt halmazon
test_loader = DataLoader(test_data,batch_size=16, shuffle=True)
evaluate(model, test_loader)

# Commented out IPython magic to ensure Python compatibility.
# ### Futtassunk mindent GPU-n!
# ### Mindent át kell pakolni a GPU memóriájába...
# 
# %%time
# NUM_EPOCHS = 10
# BATCH_SIZE = 64
# 
# #Initialize model
# model = MLP_Network(VOCAB_SIZE,HIDDEN_UNITS,OUT_CLASSES).to(device)
# 
# #Initialize optimizer
# import torch.optim as optim
# optimizer = optim.Adam(model.parameters()) # ADAM optimalizáló algoritmus
# import torch.nn as nn
# loss_fun = nn.CrossEntropyLoss().to(device)
# 
# X_train_tensor = X_train_tensor.to(device)
# Y_train_tensor = Y_train_tensor.to(device)
# train_data = TensorDataset(X_train_tensor, Y_train_tensor)
# iterator = DataLoader(train_data,batch_size=BATCH_SIZE, shuffle=True)
# 
# for i in range(NUM_EPOCHS):
#    print(i, ". epoch acc:", train(model, iterator, optimizer, loss_fun))

X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)
test_data = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_data,batch_size=16, shuffle=True)
evaluate(model, test_loader)

"""## Konvolúciós Neurális Hálózatok (CNN)

Egy ún **Konvulúciós Neurális Hálózatot** fogunk építani és tanítani a szövegosztályozási feladathoz (lásd [olvasólecke](https://www.inf.u-szeged.hu/~rfarkas/ML22/deep_learning.html)).

### Szóbeágyazások, mint jellemzőtér

A deep learning modellek ún. **szóbeágyazás**okat használnak tokenek leírására. Egy szóbeágyazás egy szóhoz egy numerikus vektort rendel. Ha két vektor közel van egymáshoz (pl. euklideszi vektortávolság szerint), akkor a két szó jelentése valamilyen értelemben hasonlít egymáshoz. Precízebben, két szóvektor akkor van közel egymáshoz ha hasonló mondatkörnyezetekben fordulnak elő. Egy fajta szóbeágyazás a [word2vec](https://towardsdatascience.com/understanding-word2vec-embedding-in-practice-3e9b8985953), de a jövő héten mélyebben megismerkedünk a beágyazásokkal...

Itt most a spacy csomagot használjuk, ami tokenizál és egy saját szóbeágyazást alkalmaz.
"""

!python -m spacy download en_core_web_lg
import spacy
nlp_en = spacy.load("en_core_web_lg")

"""A szöveget először felbontjuk szavakra, a szavakat átalakítjuk vektorokká."""

text_transform = lambda x: [word.vector for word in nlp_en(x)]

text_transform("a draught beer")

"""A `pad_sequence` segítségével a vektorokat egyenlő hosszúságúvá alakítjuk, úgy, hogy a rövidebb vektorok végére 0-kból álló vektorokat írunk."""

from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

pad_sequence([Tensor(text_transform("a draught beer")), Tensor(text_transform("a"))])

"""### Adatbetöltés
Betöltjük ismét az adatot.
"""

import pandas as pd

train_data = pd.read_csv('https://github.com/rfarkas/student_data/raw/main/sentiment/train.tsv', sep='\t')[:2000]
test_data  = pd.read_csv('https://github.com/rfarkas/student_data/raw/main/sentiment/test.tsv', sep='\t')[:700]

train_data.head()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# train_data["vecs"] = train_data["text"].apply(text_transform)
# test_data["vecs"] = test_data["text"].apply(text_transform)

"""Osztálycímkéket ne felejtsük el hozzá igazítani."""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(train_data.label)

"""A `collate_batch` függvény segítségével az adatunkat minden egyes lépésben át tudjuk alakítani."""

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch


def collate_batch(batch):
    label_list, text_list = [], []
    for row in batch:
        # Címkék átalakítása
        label_list.append(le.transform([row["label"]])[0])

        # Szöveg átalakítása
        processed_text = Tensor(row["vecs"])
        text_list.append(processed_text)

    labels_tensor = Tensor(label_list).long().to(device)

    # Szövegek egységhosszúra alakítása
    padded_vec_tensor = pad_sequence(text_list).to(device)

    return labels_tensor, padded_vec_tensor

"""A HuggingFace-es *datasets* csomag segítségével az adatot feldolgozzuk."""

!pip install datasets

from datasets import Dataset

# Dataframe-ek átalakítása Dataset-té, ami beolvasható a Data loader számára
train_dataset = Dataset.from_pandas(train_data)

test_dataset = Dataset.from_pandas(test_data)

# Data loader elkészítése, aminek megadjuk a korábban megígrt collate_batch függvényt
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              collate_fn=collate_batch)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True,
                              collate_fn=collate_batch)

print(x := next(iter(train_dataloader)))
print(x[1].shape)

"""### CNN szerkezetének megadása

Minden feladatra saját hálózatot építhetünk az egyes neuron rétegek megadásával. Ehhez egy új osztályt kell definiálni, legalább konstruktorral és forward() metódussal.
"""

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, kernel_size, output_dim):
        super().__init__()

        # utána jön a konvolúciós réteg (rétegek), itt a kernel mérete az "ablakméret"
        self.conv = nn.Conv1d(in_channels = embedding_dim,
                              out_channels = n_filters,
                              kernel_size = kernel_size)

        # végül a kimeneti réteg, ami egy egyszerű lineáris réteg
        self.fc = nn.Linear(n_filters, output_dim)

    # amikor a szöveget előrefelé ("alulról felfelé" a rétegeken át) feldolgozza a háló
    def forward(self, embedded):


        # kiolvassuk a szóbeágyazási vektorokat
        # print("embedded", embedded.size())
        # ez egy 3 dimenziós tömb (tenzor):
        # embedded = [sent len, batch size, emb dim]
        embedded = embedded.permute(1, 2, 0)
        # print("embedded_perm", embedded.size())
        #embedded = [batch size, emb dim, sent len]

        # ezután a konvolúciós réteg a RelU aktivációs függvényt használja
        conved = F.relu(self.conv(embedded))
        # print("conved", conved.size())

        # ennek a tenzornak a méretei:
        # conved = [batch size, n_filters, sent len - filter_size + 1]

        # tovább tömörítjük:
        pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
        #print("pooled", pooled.size())
        # pooled = [batch size, n_filters]

        # a háló kimenetét a lineáris réteg számolja ki
        return self.fc(pooled)

### a háló példányosítása

embedding_dim = 300
n_filters = 40
kernel_size = 3
output_dim = len(le.classes_)

model = CNN(embedding_dim, n_filters, kernel_size, output_dim)
model = model.to(device)

# a háló rétegei:
print(model)

# összesen ennyi paramétert kell tanítanunk:
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(count_parameters(model), "tanulandó változó")

model(next(iter(train_dataloader))[1])

"""### CNN tanítása

A neurális hálók tanítása egy optimalizációs feladat megoldásával törénik. Úgy akrjuk beállítani az 1.5M változót, hogy minimalizáljuk a háló kimenete és a tényleg osztálycímke közti eltérést.
"""

### A tanítás során többször végigmegyünk a tanító adatbázison (egy kör egy epoch)

def train(model, iterator, optimizer, criterion):
    # minden epoch végén ellenőrízni fogjuk az accuracyt
    epoch_acc = 0

    model.train() # inicializálás
    for batch in iterator:
        optimizer.zero_grad()
        labels = batch[0]
        vectors = batch[1]

        # predikáljuk le a tanító példákat az aktuális paraméterekkel:
        predictions = model(vectors)

        # a háló aktuális paraméterivel ennyi a hiba a batchen:
        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        # hibavisszaterjesztéssel (backpropagation) javítunk a paramétereken:
        loss.backward()
        optimizer.step()

        epoch_acc += acc.item()

    return epoch_acc / len(iterator)


def accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True) # a 3 osztályra adott kimeneti érték közül melyik a legnagyobb
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum(dtype=float) / y.shape[0]

# GPU-n akarunk tanítani:
model = CNN(embedding_dim, n_filters, kernel_size, output_dim)
model = model.to(device)

import torch.optim as optim
optimizer = optim.Adam(model.parameters()) # ADAM optimalizáló algoritmus
criterion = nn.CrossEntropyLoss() # hibafüggvény többosztályos feladatokhoz
criterion = criterion.to(device)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# ### mehet a tanítás!
# 
# NUM_EPOCHS = 20
# 
# for i in range(NUM_EPOCHS):
#     print(i, ". epoch train acc:", train(model, train_dataloader, optimizer, criterion),
#             " test acc:", evaluate(model, test_dataloader))

"""### CNN kiértékelése kiértékelő halmazon"""

### ha az epoch végén egy független validációs halmazon is ki akarjuk értékelni a modellt:

def evaluate(model, iterator):
    epoch_acc = 0
    model.eval()  # inicializálás
    with torch.no_grad():
        for batch in iterator:
            # predikció
            predictions = model(batch[1])
            # kiértékelés
            acc = accuracy(predictions, batch[0])
            epoch_acc += acc.item()

    return epoch_acc / len(iterator)

evaluate(model, test_dataloader)

"""# Gyakorló fealdatok

Futtasd az órai notebook-ot, hajtsd végre az alábbi módosításokat a rendszeren!

1. Szúrj be még egy konvolúciós réteget a hálózatba!

2. Ha a konvolúció ablakméretét 5-re állítjuk, mennyi lesz a kiértékelő halmazon a pontosság?
"""