import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud


def plot_word_cloud(wordcloud, title, ax):
    """
    Plot a word cloud on the given axis with the book name as a label.
    """
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)


def plot_side_by_side_word_cloud(quran_wordcloud, bible_wordcloud):
    """
    Plot the word clouds side by side with book labels, and combine them into one image.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot the word clouds with book names as titles
    plot_word_cloud(quran_wordcloud, "Quran Word Cloud", axes[0])
    plot_word_cloud(bible_wordcloud, "Bible Word Cloud", axes[1])

    # Add labels to the word clouds
    axes[0].set_title("Quran Word Cloud")
    axes[1].set_title("Bible Word Cloud")

    # Combine the plots into a single image
    plt.tight_layout()
    plt.show()


def plot_word_hierarchy(
        related_words_quran,
        related_words_bible,
        quran_root_word,
        bible_root_word):
    """
    Plot the hierarchy of related words using NetworkX and link the root words of both books.
    """
    G = nx.DiGraph()

    # Add the root words for Quran and Bible with labels
    G.add_node(quran_root_word, label="Quran")
    G.add_node(bible_root_word, label="Bible")

    # Add edges from the Quran hierarchy
    for level in related_words_quran:
        for word in level:
            G.add_node(word)
            G.add_edge(quran_root_word, word)

    # Add edges from the Bible hierarchy
    for level in related_words_bible:
        for word in level:
            G.add_node(word)
            G.add_edge(bible_root_word, word)

    # Link the root words of both books
    G.add_edge(quran_root_word, bible_root_word, color='red', weight=2)

    # Draw the hierarchy graph
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    plt.figure(figsize=(12, 12))

    # Draw the graph with labels, including special labels for root words
    node_labels = {
        quran_root_word: f"{quran_root_word} (Quran)",
        bible_root_word: f"{bible_root_word} (Bible)"}
    for word in G.nodes:
        if word not in node_labels:  # Add normal labels for other words
            node_labels[word] = word

    edge_colors = [G[u][v].get('color', 'black') for u, v in G.edges()]
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=node_labels,
        node_size=5000,
        node_color='skyblue',
        font_size=12,
        font_weight='bold',
        arrowsize=10,
        edge_color=edge_colors,
        width=edge_weights)

    plt.title("Word Hierarchy for Related Words (Quran & Bible)")
    plt.show()
