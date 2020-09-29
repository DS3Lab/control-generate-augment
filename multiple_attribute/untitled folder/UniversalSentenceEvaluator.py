import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

'''SCRIPT TO COMPUTE THE SENTIMENT EMBEDDINGS SIMILARITY'''


################################ FOR SENTIMENT #########################################
def process_original_data(samples=10000, classes=2):
    '''
    Selection of the samples from the orginal dataset
    :param samples: (int) number of samples we want to analyze
    :param classes: (int) number of attributes
    :return:
    '''
    data = pd.read_csv("/Users/giusepperusso/Desktop/ThesisLab 2/data/yelp_train.csv").text
    label = pd.read_csv("/Users/giusepperusso/Desktop/ThesisLab 2/data/y_yelp_train.csv").Negative
    size = samples // classes
    if classes == 2:
        data_vocab = {'Pos': [], 'Neg': []}
    for idx, (s, l) in enumerate(zip(data, label)):
        if l == 1 and len(data_vocab['Neg']) < size:
            data_vocab['Neg'].append(s)
        if l == 0 and len(data_vocab['Pos']) < size:
            data_vocab['Pos'].append(s)
        if len(data_vocab['Neg']) == size and len(data_vocab['Pos']) < size:
            break
    print(len(data_vocab['Neg']))
    sentences = np.concatenate((data_vocab['Neg'], data_vocab['Pos']))
    return sentences


def embedding_list(sentences):

    '''Compute the sentence embeddings'''
    messages = [s.split('<eos>')[0] for s in sentences]
    embedding_list = []
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(emb(messages))
    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        embedding_list.append(message_embedding)
    embedding_list = np.asarray(embedding_list)

    return embedding_list

#emb = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")
filepath = "/Users/giusepperusso/Desktop/MT-giuseppe-russo/GeneratedFolder/ExampleSentence/Sentiment10K.csv"
generated_sentences = [i for i in pd.read_csv(filepath).text]
original_sentences = process_original_data()
original_embedding = embedding_list(original_sentences)
generated_embedding = embedding_list(generated_sentences)
sim_original = cosine_similarity(original_embedding, dense_output=True)
sim_generated = cosine_similarity(generated_embedding, dense_output=True)
# generate images

import matplotlib.pyplot as plt

a = plt.imshow(sim_original)
plt.xticks((0, 5000, 10000))
plt.yticks((0, 5000, 10000))
#plt.savefig("Evaluation/Results/sentiment_similarity_orig.jpg")
plt.colorbar()
plt.show()


plt.imshow(sim_generated)
plt.xticks((0, 5000, 10000))
plt.yticks((0, 5000, 10000))
plt.colorbar()
#plt.savefig("Evaluation/Results/sentiment_similarity_gen.jpg")
plt.show()

#### qua facciamo a mano

def sentiment_matrix(sent_matrix):
    '''computing the sentiment embedding matrix'''
    nn = sent_matrix[0:5000,0:5000]
    np = sent_matrix[0:5000, 5000:10000]
    pp = sent_matrix[5000:10000, 5000:10000]
    return nn, np, pp

nn_or, np_or, pp_or = sentiment_matrix(sim_original)
nn, np, pp = sentiment_matrix(sim_generated)
or_score_matrix = [nn_or, np_or, pp_or]
gen_score_matrix = [nn, np, pp]
import numpy as np
def scoring(matrix, samples = 100):

    matrix = np.sort(matrix,axis=1)
    matrix = np.flip(matrix, axis = 1)
    return np.mean(matrix[:,0:samples], axis=1)
BIGGER_SIZE=16
SMALL_SIZE = 10
MEDIUM_SIZE=12
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)


titles = ['Neg-Neg', 'Neg-Pos', 'Pos-Pos']
for i in range(3):
    plt.gcf().subplots_adjust(bottom=0.15)
    scores_gen = scoring(gen_score_matrix[i])
    scores_or = scoring(or_score_matrix[i])
    plt.hist(scores_gen, bins=80, alpha=0.7, label="Generated Data", color="#E05E0D")
    plt.hist(scores_or, bins=80, alpha=0.7, label ="Real Data", color="#1E519E")
    plt.axvline(np.mean(scores_gen), color="#E05E0D",alpha=1, linestyle='dashed')
    plt.axvline(np.mean(scores_or), color="#1E519E",alpha=1, linestyle='dashed')
    plt.xlabel("Similarity Score")
    plt.xlim(0.2,1)
    plt.ylim(0,200)
    plt.legend()
    plt.savefig("/Users/giusepperusso/Desktop/ReportResult/"+titles[i]+".png")
    plt.show()
####################################### FOR MULTIPLE ATTRIBUTES ####################################
BIGGER_SIZE=16
SMALL_SIZE = 10
MEDIUM_SIZE=12
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
emb = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2")

#emb = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/1")
def scoring(matrix, samples = 100):

    matrix = np.sort(matrix,axis=1)
    matrix = np.flip(matrix, axis = 1)
    return np.mean(matrix[:,0:samples], axis=1)

def process_original_data(samples=10000, classes=4):
    data = pd.read_csv("//Users/giusepperusso/Desktop/MultipleAttrControl/data/TenseSentiment/yelp_train.csv").text
    label = pd.read_csv("/Users/giusepperusso/Desktop/MultipleAttrControl/data/TenseSentiment/y_yelp_train.csv")
    size = samples // classes
    if classes == 2:
        data_vocab = {'Pos': [], 'Neg': []}
    else:
        data_vocab = {'NegPres':[], 'NegPast':[], 'PosPres':[], 'PosPast':[]}
    for idx, s in enumerate(data):
        l = label[['Negative','Positive','Present','Past']].iloc[idx]
        l = l.values.tolist()
        if l == [1, 0, 1, 0] and len(data_vocab['NegPres']) < size:
            data_vocab['NegPres'].append(s)
        if l == [1, 0, 0, 1] and len(data_vocab['NegPast']) < size:
            data_vocab['NegPast'].append(s)
        if l == [0, 1, 1, 0] and len(data_vocab['PosPres']) < size:
            data_vocab['PosPres'].append(s)
        if l == [0, 1, 0, 1] and len(data_vocab['PosPast']) < size:
            data_vocab['PosPast'].append(s)
        if len(data_vocab['NegPres']) == len(data_vocab['NegPast']) == len(data_vocab['PosPres']) == len(data_vocab['PosPast']) == size:
            break
    print(len(data_vocab['NegPres']))
    sentences = np.concatenate((data_vocab['NegPres'],data_vocab['NegPast'], data_vocab['PosPres'], data_vocab['PosPast']))
    return sentences

def embedding_list(sentences):
    messages = [s.split('<eos>')[0] for s in sentences]
    embedding_list = []
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(emb(messages))
    for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
        embedding_list.append(message_embedding)
    embedding_list = np.asarray(embedding_list)

    return embedding_list

sentences = process_original_data()
filepath = "/Users/giusepperusso/Desktop/GeneratedFolder/ExampleSentence/SentTens10K.csv"
X = pd.read_csv(filepath)
generated_sentences = [i for i in pd.read_csv(filepath).text]
generated_embedding = embedding_list(generated_sentences)
#
original_sentences = process_original_data()
original_embedding = embedding_list(original_sentences)
sim_original = cosine_similarity(original_embedding, dense_output=True)
sim_generated = cosine_similarity(generated_embedding, dense_output=True)
import matplotlib.pyplot as plt
plt.imshow(sim_generated)
#plt.yticks(np.asarray((0, 125, 250,375, 500,625, 750,875, 1000))*10, ("0", "NegPres", "2500","NegPast", "5000", "PosPres", "7500", "PosPast", "10000"))
#plt.xticks(np.asarray((0, 125, 250,375, 500,625, 750,875, 1000))*10, ("0", "'NegPres'", "2500","NegPast", "5000", "PosPres", "7500", "PosPast", "10000"))
plt.yticks(np.asarray((0, 250, 500,750, 1000))*10,  ("0",  "2500", "5000",  "7500",  "10000"))
plt.xticks(np.asarray((0,  250, 500, 750, 1000))*10,  ("0",  "2500", "5000",  "7500",  "10000"))
plt.colorbar()
plt.savefig("/Users/giusepperusso/Desktop/ReportResult/Multi/sentiment_similarity_mult_gen.jpg")
plt.show()
titles = ["NegPres-NegPres", "NegPres-NegPast","NegPres-PosPres", "NegPres-PosPast",
          "NegPast-NegPast", "NegPast-PosPres","NegPast-PosPast",
          "PosPres-PosPres", "PosPres-PosPast",
          "PosPast-PosPast",]

a = plt.imshow(sim_original)
#plt.yticks(np.asarray((0, 125, 250,375, 500,625, 750,875, 1000))*10,  ("0", "NegPres", "2500","NegPast", "5000", "PosPres", "7500", "PosPast", "10000"))
#plt.xticks(np.asarray((0, 125, 250,375, 500,625, 750,875, 1000))*10,  ("0", "NegPres", "2500","NegPast", "5000", "PosPres", "7500", "PosPast", "10000"))
plt.yticks(np.asarray((0, 250, 500,750, 1000))*10,  ("0",  "2500", "5000",  "7500",  "10000"))
plt.xticks(np.asarray((0,  250, 500, 750, 1000))*10,  ("0",  "2500", "5000",  "7500",  "10000"))
plt.savefig("/Users/giusepperusso/Desktop/ReportResult/Multi/sentiment_similarity_mult_orig.jpg")
plt.colorbar()
plt.show()



def scoring(matrix, samples = 100):

    matrix = np.sort(matrix,axis=1)
    matrix = np.flip(matrix, axis = 1)
    return np.mean(matrix[:,0:samples], axis=1)

def select_cluster(matrix, limit=10000, offset = 2500, clusters=4):
    start_line = 0;
    end_line = start_line + offset;
    scores = []
    for c in range(clusters):
        if c >= 1:
            start_line+=offset; end_line+=offset
        start = start_line; end = end_line
        while(end <= limit):
            print("Cluster: {} X: {} Y: {}".format(c, (start_line,end_line), (start,end)))
            cluster = matrix[start_line:end_line, start:end]
            cluster_score = scoring(cluster,samples=limit//100)
            scores.append(cluster_score)
            start+=offset
            end+=offset

            ### qui plottiamo
    return scores

original_scores = select_cluster(sim_original,limit=10000, offset=2500)
gen_scores = select_cluster(sim_generated, limit=10000, offset=2500)

def histogram(original_scores, gen_scores):


    for i in range(len(original_scores)):
        plt.gcf().subplots_adjust(bottom=0.15)
        scores_gen = gen_scores[i]
        scores_or = original_scores[i]
        plt.hist(scores_gen, bins=80, alpha=0.7, label="Generates Data",color="#E05E0D")
        plt.hist(scores_or, bins=80, alpha=0.7, label="Real Data", color="#1E519E")
        plt.axvline(np.mean(scores_gen),  alpha=1, linestyle='dashed',color="#E05E0D")
        plt.axvline(np.mean(scores_or),  alpha=1, linestyle='dashed',color="#1E519E")
        plt.legend()
        plt.xlabel("Similarity Scores")
        plt.savefig("/Users/giusepperusso/Desktop/ReportResult/Multi/" + titles[i] + ".png")
        plt.show()
histogram(original_scores,gen_scores)
