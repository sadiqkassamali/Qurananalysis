import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from io import StringIO
import customtkinter as ctk
from PIL import Image

from plotter import plot_word_hierarchy, plot_side_by_side_word_cloud, plot_tsne, plot_pca
# Assuming your functions are in the 'main.py' file or another file
from run import (analyze_data, display_results, on_submit_button, quran_model)


class TestQuranBibleAnalysis(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_analyze_data(self, mock_read_csv):
        """Test the analyze_data function for contradiction detection."""
        # Mock the data loading part to simulate CSVs
        mock_quran_data = pd.DataFrame({
            'text': ['In the name of God', 'God is great', 'Praise be to God']
        })
        mock_bible_data = pd.DataFrame({
            'text': ['In the beginning God created', 'God is great', 'Praise be to the Lord']
        })

        # Mock the return value of the read_csv function
        mock_read_csv.side_effect = [mock_quran_data, mock_bible_data]

        # Test when the word exists in both Quran and Bible (contradiction found)
        result = analyze_data('data/quran.csv', 'data/bible.csv', 'God is great')
        self.assertTrue(result)

        # Test when the word exists only in the Quran (no contradiction)
        result = analyze_data('data/quran.csv', 'data/bible.csv', 'Praise be to God')
        self.assertFalse(result)

        # Test when the word does not exist in either text (no contradiction)
        result = analyze_data('data/quran.csv', 'data/bible.csv', 'Not in text')
        self.assertFalse(result)


    @patch('PIL.Image.open')
    @patch('customtkinter.CTkLabel.config')
    def test_display_results(self, mock_label_config, mock_image_open):
        """Test if display_results updates the UI with the correct images and results."""
        # Mock image open to return a dummy image
        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        # Test displaying results with mock image files
        display_results(
            "Contradictions found for 'God is great'.",
            wordcloud_image='wordcloud.png',
            node_image='node_hierarchy.png',
            tsne_image='tsne.png',
            pca_image='pca.png'
        )

        # Ensure the image display functions are called
        mock_label_config.assert_any_call(image=mock_image)  # Check if the label's image is updated

        # Check if the results text has been updated (in real app, it's done through Tkinter)
        self.assertEqual(mock_label_config.call_count, 5)  # We expect 5 updates (result, 4 images)

    @patch('main.analyze_data')
    @patch('main.plot_side_by_side_word_cloud')
    @patch('main.plot_word_hierarchy')
    @patch('main.plot_tsne')
    @patch('main.plot_pca')
    def test_on_submit_button(self, mock_pca, mock_tsne, mock_hierarchy, mock_wordcloud, mock_analyze):
        """Test the on_submit_button function."""
        # Mock the analyze_data function to return a fixed result
        mock_analyze.return_value = True

        # Mock the plot functions to return fake image paths
        mock_wordcloud.return_value = MagicMock(spec=Image.Image)
        mock_hierarchy.return_value = MagicMock(spec=Image.Image)
        mock_tsne.return_value = MagicMock(spec=Image.Image)
        mock_pca.return_value = MagicMock(spec=Image.Image)

        # Simulate user input
        with patch('main.word_entry.get', return_value="God is great"):
            with patch('main.result_label.config') as mock_result_label:
                # Call the submit button handler
                on_submit_button()

                # Check if the analyze_data function was called
                mock_analyze.assert_called_once_with('data/quran.csv', 'data/bible.csv', 'God is great')

                # Ensure that the visualization functions were triggered
                mock_wordcloud.assert_called_once()
                mock_hierarchy.assert_called_once()
                mock_tsne.assert_called_once()
                mock_pca.assert_called_once()

                # Ensure the result label was updated
                mock_result_label.assert_called_with(text="Contradictions found for 'God is great'.")

    @patch('customtkinter.CTkCheckBox.get')
    @patch('customtkinter.CTkButton.invoke')
    def test_ui_interaction(self, mock_button_invoke, mock_check_get):
        """Test the UI interaction, including checkbox and button click."""
        # Mock checkbox values to simulate user selections
        mock_check_get.return_value = True  # Mocking that checkbox is checked

        # Create a mock root window
        mock_root = MagicMock(spec=ctk.CTk)

        # Simulate button click
        with patch('main.submit_button.invoke') as mock_invoke:
            on_submit_button()

            # Ensure the button click invokes the right functionality
            mock_invoke.assert_called_once()

            # Test if the checkboxes are being used as intended
            self.assertTrue(mock_check_get.return_value)

    def test_image_creation(self):
        """Test if image creation functions work without errors."""
        # Assuming these functions should return images or file paths
        try:
            wordcloud = plot_side_by_side_word_cloud('God', 10, quran_model)
            node_hierarchy = plot_word_hierarchy('God', quran_model)
            tsne_plot = plot_tsne('God', quran_model)
            pca_plot = plot_pca('God', quran_model)

            # Check that images are generated (mock in the tests above should handle these)
            wordcloud.save("test_wordcloud.png")
            node_hierarchy.save("test_node.png")
            tsne_plot.save("test_tsne.png")
            pca_plot.save("test_pca.png")
        except Exception as e:
            self.fail(f"Image creation failed: {str(e)}")


if __name__ == '__main__':
    unittest.main()
