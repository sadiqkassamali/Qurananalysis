import logging

import matplotlib.pyplot as plt

from plotter import print_similar_word_cloud

logging.root.level = logging.INFO
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib

matplotlib.use('TkAgg')
import numpy as np

# Scatter plott of all words.
plt.figure(1)
circle = plt.Circle((0, 0), radius=1, fc='w')
plt.gca().add_patch(circle)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.yticks(np.arange(-3, 3, .5))
plt.xticks(np.arange(-3, 3, .5))
plt.show

# Start using the model and play with visualization
plt.figure(2)
plt.subplot(211)
# Search for work Allah and something that's closest
word = 'allah'
print_similar_word_cloud(word, 100)

plt.subplot(212)
# Search for work nuh and something that's closest
word = 'nuh'
print_similar_word_cloud(word, 100)
