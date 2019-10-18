from bert_serving.client import BertClient
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

# bert-serving-start -model_dir /Users/landon/Desktop/bertLarge -num_worker=1

bc = BertClient(check_length=False)
#
import numpy as np

while True:
    w = input("Word 1: ")
    v = input("Word 2: ")
    w_vec = np.array(bc.encode([w])[0])
    v_vec = np.array(bc.encode([v])[0])

    # count = 0
#
    # for x, y in zip(w_vec, v_vec):
        # count += (x * y)
    c = euclidean(w_vec[0], v_vec[0])
    print(f'DIst: {c}\n\n')
