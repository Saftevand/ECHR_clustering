import pickle

import visualization
import numpy as np


matrix = np.array([[0, 1, 1, 0, 0],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0]])

matrix2 = np.random.randint(4, size=(10, 10))

#matrix = None


#with open("affinity_prop_labels.pkl", "rb") as labels_pickle:
#    labels = pickle.load(labels_pickle)

#print(labels)




visualization.gui_graph_run(matrix=matrix)