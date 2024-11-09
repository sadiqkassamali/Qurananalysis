# Quran-Analysis
Quran analysis with some ML
Not sure where it will go and what will i learn from this


#Fork of quran-nlp (non arabic model and training)
https://github.com/aelbuni/quran-nlp/blob/master/README.md
> at this point there aren't any thing common, still get inspiration from above 

Quran-Analysis is an opensource project that aims to provide a boilerplate to help researchers to mine for knowledge from the Quran in english

# Usage example
# Quran and Bible Text Analysis Application

This application analyzes and compares the textual data of the Quran and Bible for contradictions, based on a given word of interest. It provides visualizations such as word clouds, word hierarchies, t-SNE, PCA, and KMeans clustering, all within a graphical user interface (GUI) built with `customtkinter`.

## Features

- **Contradiction Detection**: Analyze and detect contradictions between the Quran and Bible based on the provided word of interest.
- **Word Cloud Visualization**: Generate and display word clouds for both the Quran and Bible for the given word.
- **Word Hierarchy Visualization**: Show the hierarchy of related words for the Quran and Bible.
- **Dimensionality Reduction**:
    - **t-SNE**: Visualize the Word2Vec embeddings using t-SNE.
    - **PCA**: Visualize the Word2Vec embeddings using PCA.
- **Clustering**: Perform KMeans clustering on the Word2Vec embeddings and visualize the clusters.
- **User-friendly GUI**: Built with `customtkinter` with an easy-to-use interface to enter the word of interest and choose visualizations.

## Requirements

Ensure you have the following Python libraries installed:

```bash
pip install customtkinter pandas matplotlib seaborn wordcloud scikit-learn plotly
