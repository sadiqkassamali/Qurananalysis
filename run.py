import pandas as pd
import logging
import re
from gensim.models import Word2Vec
from plotter import plot_side_by_side_word_cloud, plot_word_hierarchy, plot_word_cloud
from wordcloud import WordCloud
import time
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(
    filename='analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_bible(file_path):
    """
    Load and preprocess Bible CSV data.
    """
    try:
        bible_df = pd.read_csv(file_path)
        bible_df.fillna('', inplace=True)
        bible_df['Text'] = bible_df['Text'].apply(
            lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))
        logging.info(f"Loaded Bible data: {len(bible_df)} verses.")
        return bible_df
    except Exception as e:
        logging.error(f"Failed to load Bible data: {e}")
        return None


def load_and_preprocess_quran(file_path):
    """
    Load and preprocess Quran CSV data.
    """
    try:
        quran_df = pd.read_csv(file_path)
        quran_df.fillna('', inplace=True)
        quran_df['Text'] = quran_df['Text'].apply(
            lambda x: re.sub(r"[^a-zA-Z\s]", "", x.lower()))
        logging.info(f"Loaded Quran data: {len(quran_df)} verses.")
        return quran_df
    except Exception as e:
        logging.error(f"Failed to load Quran data: {e}")
        return None


def train_word2vec_model(text_data):
    """
    Train a Word2Vec model on tokenized text data.
    """
    logging.info("Training Word2Vec model...")
    model = Word2Vec(
        text_data,
        vector_size=100,
        window=5,
        min_count=10,
        workers=4)
    logging.info("Word2Vec model training completed.")
    return model


def preprocess_verses_for_word2vec(df, column='Text'):
    """
    Preprocess verses for Word2Vec input: tokenizing and cleaning.
    """
    processed_verses = df[column].apply(lambda x: x.split()).tolist()
    return processed_verses


def find_related_words_hierarchy(word, model, depth=3):
    """
    Find a hierarchy of related words based on Word2Vec model.
    """
    related_words = [[word]]  # Start with the main word
    for _ in range(depth):
        next_level = []
        for w in related_words[-1]:
            try:
                similar_words = model.wv.most_similar(positive=[w], topn=5)
                next_level.extend(
                    [w[0] for w in similar_words if w[0] not in next_level])
            except KeyError:
                continue
        if next_level:
            related_words.append(next_level)
    return related_words


def generate_word_cloud(model, word_of_interest, topn=10):
    """
    Generate a word cloud based on the Word2Vec model for the given word.
    """
    similar_words = model.wv.most_similar(
        positive=[word_of_interest], topn=topn)
    word_freq = {word: score for word, score in similar_words}
    wordcloud = WordCloud(
        width=800,
        height=400).generate_from_frequencies(word_freq)
    return wordcloud


def print_similar_word_cloud(word_of_interest, topn, model):
    """
    Print the similar word cloud for the given word.
    """
    wordcloud = generate_word_cloud(model, word_of_interest, topn)
    return wordcloud

# New functions for similarity and contradiction analysis


def get_semantic_similarity(text1, text2):
    """
    Calculate semantic similarity using Sentence-BERT.
    """
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    return cosine_similarity(embeddings1, embeddings2)[0][0]


def detect_contradictions(quran_data, bible_data, threshold=0.5):
    """
    Detect contradictions based on semantic similarity.
    """
    contradictions = []
    for quran_verse, bible_verse in zip(
            quran_data['Text'], bible_data['Text']):
        similarity = get_semantic_similarity(quran_verse, bible_verse)
        if similarity < threshold:
            contradictions.append({
                'quran_verse': quran_verse,
                'bible_verse': bible_verse,
                'similarity': similarity
            })
    return contradictions


def analyze_data(
        file_path_quran=None,
        file_path_bible=None,
        word_of_interest=None,
        topn=10):
    """
    Analyze Quran and/or Bible data based on the word of interest.
    """
    quran_wordcloud = None
    bible_wordcloud = None
    quran_hierarchy = None
    bible_hierarchy = None
    quran_root_word = word_of_interest  # Root word for Quran
    bible_root_word = word_of_interest  # Root word for Bible

    if file_path_quran:
        # Load and process Quran data
        quran_data = load_and_preprocess_quran(file_path_quran)
        quran_verses = preprocess_verses_for_word2vec(quran_data)
        quran_model = train_word2vec_model(quran_verses)
        print(f"Finding similar words for '{word_of_interest}' in Quran...")
        quran_wordcloud = print_similar_word_cloud(
            word_of_interest, topn, quran_model)

        # Find related words hierarchy in Quran
        quran_hierarchy = find_related_words_hierarchy(
            word_of_interest, quran_model)
        print(
            f"Related words hierarchy in Quran for '{word_of_interest}': {quran_hierarchy}")

    if file_path_bible:
        # Load and process Bible data
        bible_data = load_and_preprocess_bible(file_path_bible)
        bible_verses = preprocess_verses_for_word2vec(bible_data)
        bible_model = train_word2vec_model(bible_verses)
        print(f"Finding similar words for '{word_of_interest}' in Bible...")
        bible_wordcloud = print_similar_word_cloud(
            word_of_interest, topn, bible_model)

        # Find related words hierarchy in Bible
        bible_hierarchy = find_related_words_hierarchy(
            word_of_interest, bible_model)
        print(
            f"Related words hierarchy in Bible for '{word_of_interest}': {bible_hierarchy}")

    # Detect contradictions between Quran and Bible
    if file_path_quran and file_path_bible:
        contradictions = detect_contradictions(quran_data, bible_data)
        print(
            f"Detected {len(contradictions)} contradictions between Quran and Bible.")

        # Log contradictions
        if contradictions:
            for contradiction in contradictions:
                logging.info(
                    f"Contradiction detected: Quran - {contradiction['quran_verse']}, Bible - {contradiction['bible_verse']}")

    # Plot the word clouds side by side with labels
    if quran_wordcloud and bible_wordcloud:
        plot_side_by_side_word_cloud(quran_wordcloud, bible_wordcloud)

    # Plot related word hierarchy and link root words
    if quran_hierarchy and bible_hierarchy:
        plot_word_hierarchy(
            quran_hierarchy,
            bible_hierarchy,
            quran_root_word,
            bible_root_word)

    # Return contradictions if necessary
    return contradictions


# Example of calling the function
if __name__ == "__main__":
    quran_file = 'data/quran.csv'  # Path to Quran CSV file
    bible_file = 'data/bible.csv'  # Path to Bible CSV file
    word_to_compare = "god"        # Example word to compare
    contradictions = analyze_data(
        file_path_quran=quran_file,
        file_path_bible=bible_file,
        word_of_interest=word_to_compare)

    if contradictions:
        print(f"Contradictions detected: {len(contradictions)}")
    else:
        print("No contradictions detected.")
