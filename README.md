# README

_Motivation_

I am faszinated by [radioWissen podcasts](https://www.br.de/mediathek/podcast/radiowissen/488) from [BR2](https://www.br.de/radio/bayern2/index.html). They provide a wide range of topics: History, literature, music, mythology, religions, philosophy, psychology, nature, biology, economy and politics. Currently (2019-10-14) there are 2068 podcasts online. Therefore it is sometimes hard to find podcasts that I want to listen to based on quite specific query. Since I like the idea of embeddings to represent content, I have written a function to inspire me what I can listen to next.

The embeddings are based on feed-forward Neural-Net Language Models with 128-dimensional embedding vectors from [tfhub.dev](https://tfhub.dev/google/nnlm-de-dim128-with-normalization/1).

# Usage of radioWissen search function (radioWissen usage)

The idea is to get an inspiration which podcasts -- with similar content -- are available.

Find `top_n` closest podcasts to `search_string`. 

```bash
$ python quick_search_radiowissen.py --search_string="Klassizismus" --top_n=15
Searching for best matches...
                                                    Klassizismus
Gottfried Semper - Architekt und Revolutionär           0.731763
Hieronymus Bosch - Im Lustgarten der Dämonen            0.691967
Humanismus, Renaissance, Reformation - Aufbruch...      0.679882
Béla Bartók- Meister der Moderne                        0.666955
Henri de Toulouse-Lautrec - Genie der Plakatkunst       0.666286
Igor Strawinsky - Der Schock der Moderne                0.662641
Architekt und Bauhausgründer - Walter Gropius           0.661809
Jugendstil - Natur als Kunst, Schönheit als Rev...      0.657266
Ludwig II. - Der Mondkönig                              0.654857
Merkantilismus - Zölle für den Adelsprunk               0.649388
Versailles - Musik am Hof des Sonnenkönigs              0.649136
Renaissance-Gärten - Kunstwerk aus Himmel, Erde...      0.648772
Herrlichkeit und Katzenjammer - Die Epoche der ...      0.648647
Der Englische Garten - Münchens grünes Herz             0.646806
Das frühe Byzanz - Ein christliches Imperium im...      0.645438
```

Most of the results are related to architecture or at least art. 


# Demo (general usage)

Use the code for your own data.

Find clostest match from list of sentences (or list of single words) to a search query.

### 1. Imports

```python
#Suppress tensorflow infos, deprecation warnings
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(1000)
tf.__version__ # Tested with '1.13.1', does not work with 2.x
# 
import embeddings as es
```

### 2. Get Embeddings

Embed sentences or a list of words.

```python
e = es.TfHubEmbedder()
data = ["Auto", "Bus", "Caesar"]
e.embed(data)
print("Embeddings' shape: {}".format(e.embeddings.shape))
e.save("data/demo_emb.npy", "data/demo_titles.npy")
e.load("data/demo_emb.npy", "data/demo_titles.npy")
> Embeddings' shape: (3, 128)
```

### 3. Find ClosestEmbedding

Compare embeddings from (2) with your `search_string`.

```python
ce = es.ClosestEmbedding(e.embeddings, e.titles)
sim_df = ce.find_closest(search_string="VW")
ce.plot_result()
print(sim_df)
>               VW
> Auto    0.720517
> Bus     0.602985
> Caesar  0.468566

```
