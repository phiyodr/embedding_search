import warnings
warnings.filterwarnings("ignore")

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import pandas as pd
#from IPython.display import HTML

from typing import List
import typing
ListOfStrings = List[str]

# Embedder

import warnings
warnings.filterwarnings("ignore")
tf.logging.set_verbosity(1000)

class TfHubEmbedder():
    """Embed list of strings to latent vector representation using models from tfhub.dev."""
    def __init__(self, model_name: str='https://tfhub.dev/google/nnlm-de-dim128-with-normalization/1'):
        self.model = hub.Module(model_name)
        self.embeddings = None
        self.titles = None
        
    def embed(self, x: ListOfStrings, titles: ListOfStrings = None):
        """
        Args:
            x: List of sentences or 
        """
        assert isinstance(x, typing.List)
        embeddings = self.model(x)
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(tf.initialize_all_variables())
            emb = sess.run(embeddings)
        self.embeddings = emb
        if titles:
            self.titles = titles
        else:
            self.titles = x
    
    def save(self, file_name: str, titles_file_name: str=None):
        """Save embedding and titles optionally."""
        np.save(file_name, self.embeddings) 
        if titles_file_name:
            np.save(titles_file_name, self.titles) 
            
    def save_for_projector(self, embeddings_file_name="embeddings.tsv", meta_file_name="meta.tsv", 
                           meta_file_column_names=None, topic_list=None):
        """Save embeddings and metafile to visualize data on http://projector.tensorflow.org/."""
        #np.savetxt(self.embeddings, x, delimiter="\t")
        #df[['title', 'topic']].to_csv(meta_file, sep="\t", index=False)
        pass
    
    def load(self, file_name, titles_file_name=None):
        self.embeddings = np.load(file_name)
        if titles_file_name:
            self.titles = np.load(titles_file_name)

# ClosestEmbedding

class ClosestEmbedding:
    
    def __init__(self, embeddings, titles, model_name='https://tfhub.dev/google/nnlm-de-dim128-with-normalization/1', 
            embeddings_file_name=None, titles_file_name=None):
        self.embeddings = embeddings
        self.titles = titles
        self.embedder = TfHubEmbedder(model_name)
        self.ego_embedding = None
        self.ego_name = None
        self.result = None
        
    def find_closest(self, search_string: str, top_n: int=None, normalize: bool=True):
        self.ego_name = search_string

        # Set top_n
        embeddings_size = self.embeddings.shape[0]
        if top_n == None:
            top_n = embeddings_size
        elif top_n > embeddings_size:
            top_n = embeddings_size
            
        # Ego embedding 
        self.embedder.embed([search_string])
        self.ego_embedding = self.embedder.embeddings

        # Cosine sim
        sim = 1 - pairwise_distances(self.embeddings, self.ego_embedding, metric="cosine")
        
        # As DataFrame
        sim_df = pd.DataFrame(sim)
        sim_df.index = self.titles
        sim_df = pd.DataFrame(sim_df.nlargest(top_n, 0))
        sim_df.columns = [search_string]
        
        # Normalize
        if normalize:
            sim_df = sim_df.apply(lambda x: (x+1) / 2)

        self.result = sim_df
        return sim_df


    def plot_result(self):
        ax = self.result.sort_values(self.result.columns[0]).plot.barh()
        ax.set_xlim(0, 1)
        ax.axvline(x=.5, color ="red", linestyle='--', lw=2)
