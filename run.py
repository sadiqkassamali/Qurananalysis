import logging
import matplotlib.pyplot as plt
from plotter import print_similar_word_cloud  # Ensure this is defined correctly
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import nltk
import string
import os

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Set logging level
logging.root.level = logging.INFO

# Function to load and preprocess text
def load_and_preprocess(file_path):
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        
    def preprocess_text(text):
        text = text.lower()  # Lowercase
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = text.split()  # Tokenize
        stop_words = set(stopwords.words('english'))  # Stop words
        words = [word for word in words if word not in stop_words]  # Remove stop words
        return words
    
    return preprocess_text(text)

# Function to create and train Word2Vec model
def create_word2vec_model(words, vector_size=100, window=5, min_count=1):
    model = Word2Vec([words], vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

# Function to find similar words
def find_similar_words(word, model, top_n=10):
    if word in model.wv:
        return model.wv.most_similar(word, topn=top_n)
    else:
        logging.warning(f"Word not found in model: {word}")
        return []

# Function to plot dendrogram
def plot_dendrogram(embeddings, labels):
    linked = linkage(embeddings, 'ward')
    plt.figure(figsize=(10, 7))
    dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram of Similar Words')
    plt.xlabel('Words')
    plt.ylabel('Distance')
    plt.show()

# Main analysis function
def analyze_texts(file_paths, word_of_interest):
    # Load and preprocess text
    words_list = [load_and_preprocess(file) for file in file_paths]

    # Combine all words into a single list for Word2Vec
    combined_words = []
    for words in words_list:
        combined_words.extend(words)

    # Train Word2Vec model
    model = create_word2vec_model(combined_words)

    # Find similarities for the word of interest
    similar_words = find_similar_words(word_of_interest, model)

    # Prepare embeddings for clustering
    embeddings = np.array([model.wv[word] for word, _ in similar_words if word in model.wv])

    # Plot dendrogram
    plot_dendrogram(embeddings, [word for word, _ in similar_words])

    # Visualize the results with word cloud
    plt.figure(figsize=(10, 7))
    plt.title(f'Similar words for "{word_of_interest}"')
    print_similar_word_cloud(word_of_interest, 100)  # Modify if needed for the model's context
    plt.show()

    # Display similar words as text
    print("Similar words:", similar_words)

# Usage
if __name__ == "__main__":
    # Example file paths
    file_paths = ['quran.txt', 'bible.txt']  # Change these to your actual file paths
    word_of_interest = 'peace'  # Change to the word you want to analyze
    
    analyze_texts(file_paths, word_of_interest)
