import click
import pandas as pd
import embeddings
import warnings
warnings.filterwarnings("ignore")
#import tensorflow as tf
#tf.logging.set_verbosity(1000)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

@click.command()
@click.option('--search_string', help='The topic you are interested in.')
@click.option('--radiowissen_embeddings_file', default="data/radiowissen_embeddings.npy", help='File name of RadioWissen embeddings.')
@click.option('--radiowissen_titles_file', default="data/radiowissen_titles.npy", help='File name of RadioWissen titles.')
@click.option('--top_n', default=20, help='Return n best fits.')
def quick_search_radiowissen(search_string, radiowissen_embeddings_file, radiowissen_titles_file, top_n=20):
	e = embeddings.TfHubEmbedder()
	e.load(radiowissen_embeddings_file, radiowissen_titles_file)
	ce = embeddings.ClosestEmbedding(e.embeddings, e.titles)
	sim_df = ce.find_closest(search_string, top_n)
	print(sim_df)
	return sim_df

if __name__ == '__main__':
	print("Searching for best matches...")
	sim_df = quick_search_radiowissen()