import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px


def plot_side_by_side_word_cloud(wordcloud1, wordcloud2, title1='Quran Word Cloud', title2='Bible Word Cloud'):
    """
    Plot two WordClouds side by side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(wordcloud1, interpolation="bilinear")
    axs[0].set_title(title1)
    axs[0].axis('off')
    axs[1].imshow(wordcloud2, interpolation="bilinear")
    axs[1].set_title(title2)
    axs[1].axis('off')
    plt.show()


def plot_word_hierarchy(quran_hierarchy, bible_hierarchy, quran_root_word, bible_root_word):
    """
    Plot a word hierarchy as a tree structure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Word Hierarchy for "{quran_root_word}" and "{bible_root_word}"')
    ax.plot(range(len(quran_hierarchy)), quran_hierarchy, label=f'{quran_root_word} Hierarchy')
    ax.plot(range(len(bible_hierarchy)), bible_hierarchy, label=f'{bible_root_word} Hierarchy')
    ax.legend()
    plt.show()


def plot_word_cloud(wordcloud):
    """
    Plot a single word cloud.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def plot_tsne(embeddings, labels):
    """
    Plot the t-SNE of the Word2Vec embeddings.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    df = pd.DataFrame(tsne_results, columns=['x', 'y'])
    df['label'] = labels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='x', y='y', hue='label', data=df, palette='viridis', legend='full')
    plt.title('t-SNE visualization of Word2Vec Embeddings')
    plt.show()


def plot_pca(embeddings, labels):
    """
    Plot the PCA of the Word2Vec embeddings.
    """
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(embeddings)

    df = pd.DataFrame(pca_results, columns=['x', 'y'])
    df['label'] = labels

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='x', y='y', hue='label', data=df, palette='viridis', legend='full')
    plt.title('PCA visualization of Word2Vec Embeddings')
    plt.show()


def plot_kmeans_clustering(embeddings, n_clusters=5):
    """
    Plot KMeans clustering of the Word2Vec embeddings.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    df = pd.DataFrame(embeddings, columns=[f'feature_{i}' for i in range(embeddings.shape[1])])
    df['cluster'] = clusters

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df.columns[0], y=df.columns[1], hue='cluster', data=df, palette='tab10')
    plt.title(f'KMeans Clustering of Word2Vec Embeddings (n_clusters={n_clusters})')
    plt.show()


def plot_heatmap(similarity_matrix):
    """
    Plot a heatmap of the similarity matrix (e.g., Word2Vec cosine similarity).
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Heatmap of Semantic Similarities')
    plt.show()


def plot_dendrogram(embeddings, labels):
    """
    Plot a dendrogram based on hierarchical clustering of embeddings.
    """
    Z = linkage(embeddings, 'ward')

    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title('Dendrogram of Word2Vec Embeddings')
    plt.show()


def plot_bar(data, title='Bar Plot', xlabel='Categories', ylabel='Values'):
    """
    Simple bar plot for categorical data.
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_interactive_bar(data, title='Interactive Bar Plot', xlabel='Categories', ylabel='Values'):
    """
    Interactive bar plot using Plotly for better interactivity.
    """
    fig = px.bar(data, x=data.index, y=data.values, title=title, labels={data.index: xlabel, data.values: ylabel})
    fig.show()


def plot_choropleth(data, locations, color='red'):
    """
    Plot a choropleth map (geographical data visualization).
    """
    fig = px.choropleth(data_frame=data, locations=locations, color=color)
    fig.show()
