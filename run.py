import logging
import matplotlib.pyplot as plt

logging.root.level = logging.INFO
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import re
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

# Start of progran
d = pd.read_csv('data\quran.csv')
if (len(d) > 0):
    print("print total rows count: ")
    print(len(d))
    # print("print total data before NaN removal: ")
    # print(d)
    d.fillna('', inplace=True)
    concoc = d['Surah'].astype(str) + "-" + d['Ayah'].astype(str)

    # print("print total data after NaN removal: " )
    # print(d)
    # print(concoc)
    concoc = pd.DataFrame(concoc)
    dropped_column = d.drop(d.columns[-1], axis=1)  # Drop last column
    print("print total data After last column removal: ")
    print(dropped_column)
    dropped_column = pd.DataFrame(dropped_column)
    # print(dropped_column)
    # print(dropped_column.head()) #if you want to view first n-rows
    # print(dropped_column.tail())  # if you want to view last  n-rows
    # print(dropped_column[20:30 + 1])  #slice and dice of rows from 20 : 30
    # print(dropped_column.describe()) #Statistics of your dataframe if value is int

# spliting each word
if range(len(dropped_column['Text'])):
    dropped_column['Text'] = dropped_column['Text'].str.split()
print("Splitting each word in Text : ")
print(dropped_column)

SurahAyah = concoc.values.tolist()
print(SurahAyah)
for i in range(len(SurahAyah)):
    SurahAyah[i] = [number.lower() for number in SurahAyah[i] if re.match('^[0-9-]+', number)]
print(SurahAyah)
# You can filter for one surah too if you want!
verses = dropped_column['Text'].values.tolist()
print(verses)
for i in range(len(verses)):
    verses[i] = [word.lower() for word in verses[i] if re.match('^[a-zA-Z]+', word)]

# Creating a model verses

model = Word2Vec(verses, min_count=5, window=2, workers=20, alpha=0.22)
model.wv.vocab
model.save("quran.bin")
X = model[model.wv.vocab]
pca = PCA(n_components=2)
pca.fit(X)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))

circle = plt.Circle((0, 0), radius=1, fc='w')
plt.gca().add_patch(circle)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.yticks(pd.np.arange(-3, 3, .25))
plt.xticks(pd.np.arange(-3, 3, .25))
plt.show()
