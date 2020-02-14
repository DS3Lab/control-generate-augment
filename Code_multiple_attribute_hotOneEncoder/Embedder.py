import tqdm
import numpy as np
import torch
import torch.nn as nn

class Embedder:

    def __init__(self, filepath):
        self.filepath = filepath
        self.embedding = self.uploading_glove()

    def uploading_glove(self):
        GLOVE_FILENAME = self.filepath
        glove_index = {}
        n_lines = sum(1 for line in open(GLOVE_FILENAME))
        with open(GLOVE_FILENAME) as fp:
            for line in tqdm.tqdm(fp, total=n_lines):
                split = line.split()
                word = split[0]
                vector = np.array(split[1:]).astype(float)
                glove_index[word] = vector

        return glove_index

    def build_weight_matrix(self, vocabulary, embedding_size):
        weights_matrix = np.zeros((len(vocabulary), embedding_size))
        count_missing = 0
        count_found = 0
        for i, word in enumerate(vocabulary):
            try:
                weights_matrix[i] = self.embedding[word]
                count_found+=1
                #print("Found_word: ", word)
            except KeyError:
                #print("Missing word: ",word)
                if word == '<unk>':
                    weights_matrix[i] = np.full((embedding_size,),2)
                if word == '<sos>':
                    weights_matrix[i] = np.ones((embedding_size,))
                if word == '<pad>':
                    weights_matrix[i] = np.zeros((embedding_size,))
                if word == '<eos>':
                    weights_matrix[i] = np.full((embedding_size,), 5)
                else:
                    count_missing += 1
                  #  print(word)
                    weights_matrix[i] = np.random.rand(embedding_size)
        #print("missing: ", count_missing, "found: ", count_found)
        return weights_matrix

    def create_embedding_layer(self, weight_matrix, non_trainable=True):
        weight_matrix = torch.from_numpy(weight_matrix)
        num_embeddings, embedding_dim = weight_matrix.size()
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': weight_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False

        return emb_layer, num_embeddings, embedding_dim
'''
emb_size = 50
word2idx = {'queen': 5, 'king':6}
embedding_dir = "glove/glove.6B."+str(emb_size)+"d.txt"
glove = Embedder(filepath=embedding_dir)
embedding_mtx = glove.build_weight_matrix(word2idx.keys(), emb_size)
emb_layer, num_embeddings, embedding_dim = glove.create_embedding_layer(weight_matrix=embedding_mtx)
'''