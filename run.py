import customtkinter as ctk
import logging
from PIL import Image, ImageTk
from plotter import plot_side_by_side_word_cloud, plot_word_hierarchy, plot_tsne, plot_pca, plot_kmeans_clustering
import pandas as pd  # Assuming pandas for loading CSV data

# Set up logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
console_logger = logging.getLogger()

# Placeholder function for analyzing data (Quran and Bible text)
def analyze_data(file_path_quran, file_path_bible, word_of_interest):
    # In a real implementation, this would analyze the data for contradictions.
    # Here, we'll just return a mock result.
    quran_data = pd.read_csv(file_path_quran)
    bible_data = pd.read_csv(file_path_bible)

    # Mock contradiction detection logic based on word_of_interest
    contradictions_found = word_of_interest.lower() in quran_data['text'].str.lower().values and \
                           word_of_interest.lower() in bible_data['text'].str.lower().values

    return contradictions_found

# Placeholder variable for the Quran model (can be your actual dataset or a model)
quran_model = pd.read_csv('data/quran.csv')  # Example of loading the Quran data into a pandas DataFrame

def display_results(results_text, wordcloud_image=None, node_image=None, tsne_image=None, pca_image=None):
    """Update the UI with results and images."""
    result_label.config(text=results_text)

    # Display Word Cloud Image
    if wordcloud_image:
        wordcloud_img = Image.open(wordcloud_image)
        wordcloud_img = wordcloud_img.resize((400, 200))
        wordcloud_tk = ImageTk.PhotoImage(wordcloud_img)
        wordcloud_label.config(image=wordcloud_tk)
        wordcloud_label.image = wordcloud_tk
    else:
        wordcloud_label.config(image=None)

    # Display Node Hierarchy Image
    if node_image:
        node_img = Image.open(node_image)
        node_img = node_img.resize((400, 200))
        node_tk = ImageTk.PhotoImage(node_img)
        node_label.config(image=node_tk)
        node_label.image = node_tk
    else:
        node_label.config(image=None)

    # Display t-SNE Image
    if tsne_image:
        tsne_img = Image.open(tsne_image)
        tsne_img = tsne_img.resize((400, 200))
        tsne_tk = ImageTk.PhotoImage(tsne_img)
        tsne_label.config(image=tsne_tk)
        tsne_label.image = tsne_tk
    else:
        tsne_label.config(image=None)

    # Display PCA Image
    if pca_image:
        pca_img = Image.open(pca_image)
        pca_img = pca_img.resize((400, 200))
        pca_tk = ImageTk.PhotoImage(pca_img)
        pca_label.config(image=pca_tk)
        pca_label.image = pca_tk
    else:
        pca_label.config(image=None)

def on_submit_button():
    word_of_interest = word_entry.get()  # Get user input
    quran_file = 'data/quran.csv'  # Path to Quran CSV file
    bible_file = 'data/bible.csv'  # Path to Bible CSV file

    # Perform analysis and get results
    logging.info(f"Starting analysis for word: {word_of_interest}")
    contradictions = analyze_data(file_path_quran=quran_file, file_path_bible=bible_file, word_of_interest=word_of_interest)

    # Create the results text
    if contradictions:
        results_text = f"Contradictions found for '{word_of_interest}'. Check the logs for details."
    else:
        results_text = f"No contradictions found for '{word_of_interest}'."

    # Get visualization choices from UI options
    show_wordcloud = wordcloud_var.get()
    show_node = node_var.get()
    show_tsne = tsne_var.get()
    show_pca = pca_var.get()

    wordcloud_image = None
    node_image = None
    tsne_image = None
    pca_image = None

    if show_wordcloud:
        logging.info("Generating Word Cloud visualization...")
        wordcloud = plot_side_by_side_word_cloud(word_of_interest, 10, quran_model)
        wordcloud_path = 'wordcloud.png'
        wordcloud.to_file(wordcloud_path)
        wordcloud_image = wordcloud_path

    if show_node:
        logging.info("Generating Node Hierarchy visualization...")
        node_hierarchy = plot_word_hierarchy(word_of_interest, quran_model)
        node_hierarchy_path = 'node_hierarchy.png'
        node_hierarchy.to_file(node_hierarchy_path)
        node_image = node_hierarchy_path

    if show_tsne:
        logging.info("Generating t-SNE visualization...")
        tsne_plot = plot_tsne(word_of_interest, quran_model)
        tsne_path = 'tsne.png'
        tsne_plot.to_file(tsne_path)
        tsne_image = tsne_path

    if show_pca:
        logging.info("Generating PCA visualization...")
        pca_plot = plot_pca(word_of_interest, quran_model)
        pca_path = 'pca.png'
        pca_plot.to_file(pca_path)
        pca_image = pca_path

    # Display the results and images in the UI
    display_results(results_text, wordcloud_image, node_image, tsne_image, pca_image)

# Initialize main UI window
root = ctk.CTk()

root.title("Quran and Bible Analysis")
root.geometry("1000x800")

# Set dark theme
ctk.set_appearance_mode("dark")

# Label for user question input
question_label = ctk.CTkLabel(root, text="Enter Word or Question:")
question_label.pack(padx=10, pady=10)

# Entry for user input
word_entry = ctk.CTkEntry(root, width=400)
word_entry.pack(padx=10, pady=10)

# Option to select which visualization(s) to show
visualization_label = ctk.CTkLabel(root, text="Choose Visualizations:")
visualization_label.pack(padx=10, pady=10)

# Checkboxes for different visualizations
wordcloud_var = ctk.BooleanVar(value=True)
node_var = ctk.BooleanVar(value=False)
tsne_var = ctk.BooleanVar(value=False)
pca_var = ctk.BooleanVar(value=False)

wordcloud_check = ctk.CTkCheckBox(root, text="Word Cloud", variable=wordcloud_var)
wordcloud_check.pack(padx=10, pady=5)

node_check = ctk.CTkCheckBox(root, text="Node Hierarchy", variable=node_var)
node_check.pack(padx=10, pady=5)

tsne_check = ctk.CTkCheckBox(root, text="t-SNE", variable=tsne_var)
tsne_check.pack(padx=10, pady=5)

pca_check = ctk.CTkCheckBox(root, text="PCA", variable=pca_var)
pca_check.pack(padx=10, pady=5)

# Submit button to start analysis
submit_button = ctk.CTkButton(root, text="Analyze", command=on_submit_button)
submit_button.pack(pady=20)

# Result Label to display contradictions or analysis output
result_label = ctk.CTkLabel(root, text="", wraplength=700, justify="left")
result_label.pack(padx=10, pady=10)

# Labels to display images
wordcloud_label = ctk.CTkLabel(root)
wordcloud_label.pack(padx=10, pady=10)

node_label = ctk.CTkLabel(root)
node_label.pack(padx=10, pady=10)

tsne_label = ctk.CTkLabel(root)
tsne_label.pack(padx=10, pady=10)

pca_label = ctk.CTkLabel(root)
pca_label.pack(padx=10, pady=10)

# Run the application
root.mainloop()
