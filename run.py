import logging

import matplotlib.pyplot as plt

from plotter import print_similar_word_cloud

logging.root.level = logging.INFO
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
import matplotlib

matplotlib.use('TkAgg')

'''
#Scatter plott of all words.
circle = plt.Circle((0, 0), radius=1, fc='w')
plt.gca().add_patch(circle)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.yticks(pd.np.arange(-3, 3, .25))
plt.xticks(pd.np.arange(-3, 3, .25))
plt.show()
'''
# Start using the model and play with visualization
plt.figure(1)
plt.subplot(211)
# Search for work Prophet and something that's closest
word = 'prophet'
print_similar_word_cloud(word, 20)
plt.subplot(212)
# Search for work moses and something that's closest
word = 'allah'
print_similar_word_cloud(word, 20)
