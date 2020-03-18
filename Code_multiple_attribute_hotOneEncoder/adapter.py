import pandas as pd
import numpy as np
import json
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import string
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
import spacy
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from langdetect import detect
###### PARAMS ######

class TenseDataset:

    def __init__(self, dataset, n_class):
        self.dataset = dataset
        self.n_class = n_class
        self.nlp = spacy.load("en_core_web_sm")

    def pos(self, sentence):
        sentence = sent_tokenize(sentence)[0]
        doc = self.nlp(sentence)
        present = 0
        past = 0
        for idx, token in enumerate(doc):
            if token.tag_ == "VBP" or token.tag_ == "VBZ":
                present += 1
            if token.tag_ == "VBD":
                past += 1
        if present > past:
            return "pres", sentence
        if past > present:
            return "pas", sentence
        else:
            return "no", sentence

    def labeling(self):

        labels = []
        dataset = []
        for i in range(self.dataset.shape[0]):
            print(i)
            sentence = self.dataset.text.iloc[i]
            tag, sentence = self.pos(sentence)
            if tag == "pres":
                labels.append(0)
            if tag == "pas":
                labels.append(1)
            if tag == 'no':
                labels.append(-1)

        return labels
class SentimentDataset:

    def __init__(self, dataset, n_class):

        self.dataset = dataset
        self.n_class = n_class

    # finding the sentences which are positive and the sentence which are negative
    def splitting(self):
        labels = self.dataset['stars'].values

        l_pos = np.sort(np.concatenate((np.where(labels == 4)[0], np.where(labels == 5)[0])))
        l_neg = np.sort(np.concatenate((np.where(labels == 0)[0], np.where(labels == 1)[0], np.where(labels == 2)[0])))
        l_neu = np.sort(np.where(labels == 3)[0])

        pos = self.dataset.loc[l_pos].reset_index().drop(['index'], axis=1)
        neg = self.dataset.loc[l_neg].reset_index().drop(['index'], axis=1)
        neu = self.dataset.loc[l_neu].reset_index().drop(['index'], axis=1)

        if self.n_class == 3:
            self.summary = {'pos': len(l_pos),
                            'neg': len(l_neg),
                            'neu': len(l_neu)}

        else:
            self.summary = {'pos': len(l_pos),
                            'neg': len(l_neg),
                            }
            self.datasets = {'pos': pos, 'neg': neg}

    # labeling with vader
    def soft_labeling(self, dataset, assignment="soft", compatibility=None):

        '''

        :param data: sentences
        :param assignment: if 'soft' only compound values, else categorical value
        :param compatibility: default is None, in case assigmnent is ''hard' compatibility check if there are misslabelled data in vader assigment and remove them
        :return: labels or (labels, cleaned data)
        '''

        labels = []
        sid = SentimentIntensityAnalyzer()
        data = dataset['text']
        # data are already the sentences
        if assignment == 'soft':
            for idx, i in enumerate(data):
                score = sid.polarity_scores(i)['compound']
                labels.append(score)
            labels = pd.DataFrame({'stars': labels})
            dataset.stars = labels

        if assignment == 'hard':
            for idx, i in enumerate(data):

                score = sid.polarity_scores(i)['compound']
                if score >= 0.05:
                    labels.append(1)
                if score <= -0.05:
                    labels.append(0)
                if score > -0.05 and score < 0.05:
                    labels.append(2)
            labels = np.asarray(labels)
            idx = np.where(labels == compatibility)[0]
            labels = labels[idx]
            print("idx: ", idx)
            labels = pd.DataFrame({'stars': labels})
            if len(idx) > 0:
                dataset = dataset.loc[idx].reset_index().drop(['index'], axis=1)
            dataset.stars = labels

        return dataset

    # all kinds of laveling
    def labeling(self, assigment):

        global pos_l, neg_l, neu_l
        self.splitting()
        pos, neg = self.datasets['pos'], self.datasets['neg']

        # labeling
        if assigment == 'hard':
            pos = pos.replace({4: 1, 5: 1});
            neg = neg.replace({0: 0, 1: 0, 2: 0});
            if self.n_class > 2:
                neu = self.datasets['neu'].replace({3: 2})

        if assigment == 'soft':
            pos, neg = self.soft_labeling(pos), self.soft_labeling(neg)
            if self.n_class > 2:
                neu = self.soft_labeling(self.datasets['neu'])

        if assigment == 'soft_hard':
            pos = self.soft_labeling(pos, assignment='hard', compatibility=1)
            neg = self.soft_labeling(neg, assignment='hard', compatibility=0)
            if self.n_class > 2:
                neu = self.soft_labeling(self.datasets['neu'], assignment='hard', compatibility=2)

        if self.n_class == 2:
            self.datasets = {'pos': pos,
                             'neg': neg
                             }
        if self.n_class == 3:
            self.datasets = {'pos': pos,
                             'neg': neg,
                             'neu': neu
                             }

    def remove_key(self, dictionary, key):
        r = dict(dictionary)
        del r[key]
        return r

    def assembling(self, assignment):

        self.labeling(assignment)
        min_key = min(self.summary)
        min_dataset = self.datasets[min_key]
        if self.n_class == 2:
            summary = self.remove_key(self.summary, min_key)
            max_key = [*summary][0]
            max_dataset = self.datasets[max_key]
            idx = np.arange((int(min_dataset.shape[0]*1.25)))
            print("Minority Class: {} Majority Class: {}".format(min_dataset.shape[0], len(idx)))
            max_dataset = max_dataset.loc[idx]
            dataset = pd.concat((min_dataset, max_dataset))
            return dataset
        else:
            summary = self.remove_key(self.summary, min_key)
            middle_key = min(summary)
            middle_dataset = self.datasets[middle_key]
            summary = self.remove_key(self.summary, middle_key)
            max_key = [*summary][0]
            max_dataset = self.datasets[max_key]
            idx = np.arange((middle_dataset.shape[0] + max_dataset.shape[0])//2)
            dataset = pd.concat((min_dataset, middle_dataset, max_dataset.loc[idx]))
            return dataset.reset_index().drop(['index'], axis=1)


class DatasetAdapter:
    def __init__(self, datapath, samples, data_review, threshold, n_classes, assignment, category='restaurant', storing=None):
        '''

        :param datapath: where yelp is stored
        :param samples: how many samples we want to start with
        :param data_review: json file which helps us in findinf the difference business
        :param sent_thr: maximum sentence length of our reviews
        :param n_classes: Dictionary which contains how many classes per attributesù
        :param assignment: string 'hard' (categorical based on ratings), 'soft' (continuous based on vader), 'soft_hard'(categorical based on vader)
        :param category: bussiness we are interested. Default is 'restaurant
        '''
        if samples > 0:

            self.dataset = pd.read_csv(datapath, nrows=samples)
        else:
            self.dataset = pd.read_csv(datapath)
        self.category = category
        self.review_data = []
        self.sent_thr = threshold['sentence']
        self.word_thr = threshold['word']
        self.storing_path = storing['path']
        self.test_size = storing['test_size']
        for line in open(data_review, 'r'):
            self.review_data.append(json.loads(line))
        self.n_classes = n_classes
        self.assignment = assignment
        if category == 'restaurant':
            self.restaurant_codes()
        self.tokens_numb = []
        self.samples = samples


    def adaptation(self, loading=False, sentiment_dataset=None):

        if loading==False:
            print("1: find resturants")

            restaurants = self.find_restaurants()
            restaurants.to_csv("total_restaurants.csv", index=False)
            return restaurants
            sentimentBuilder = SentimentDataset(self.restaurants, self.n_classes['sentiment'])
            print("2: Assembling sentiment dataset")

            sentiment_dataset = sentimentBuilder.assembling(self.assignment) # the sentiment dataset is the referenced one. On this one we compute all the necessary infomation
            sentiment_dataset = sentiment_dataset.reset_index().drop(['index'], axis=1)
            sentiment_dataset.to_csv("sentimet_data/sentiment" + str(self.samples) + ".csv", index=False)
            #sentiment_dataset.to_csv("sentiment_data/sentiment.csv", index=False)
            return "end"
        print("3: Cleaning sentiment dataset")
        sentences = self.preprocessing(sentiment_dataset)
        print("sentences_shape: ", sentences.shape[0])
        sentiment_dataset.text = sentences
        #sentiment_dataset.to_csv("sentimet_data/sentiment_proc.csv", index=False)

        print("4: Cleaning languages")

        sentiment_dataset = self.language_cleaning(sentiment_dataset)
        sentiment_dataset.to_csv("language.csv", index=False)

        tenseBuilder = TenseDataset(dataset=sentiment_dataset, n_class=2)
        labels = tenseBuilder.labeling()
        print("5: filtering the data")
        drop_idx = np.where(np.asarray(labels) == -1)[0]
        labels = pd.DataFrame({'tense': labels})
        labels.to_csv("sentimet_data/labels.csv", index=False)
        dataset = pd.concat((sentiment_dataset, labels), axis=1)
        dataset = dataset.drop(drop_idx)
        labels = self.labels_mapping(dataset[['stars','tense']])

        dataset = dataset.reset_index().drop(['index'], axis= 1)
        labels = labels.reset_index().drop(['index'], axis= 1)


        return dataset, labels

    def storing(self, X, y):
        X, y = shuffle(X, y, random_state=42)
        X_train, X_val, X_train, X_val = train_test_split(X, y, test_size=self.test_size, random_state=42)
        X_train.to_csv(self.storing_path + "yelp_train.csv", index=False)
        X_val.to_csv(self.storing_path + "yelp_valid.csv", index=False)
        X_train.to_csv(self.storing_path + "y_yelp_train.csv", index=False)
        X_val.to_csv(self.storing_path + "y_yelp_valid.csv", index=False)

    def labels_mapping(self, labels):

        #for now it is just two classes

        y = []
        for i, j in zip(labels.stars, labels.tense):
            if i == 0 and j == 0:
                y.append(0)
            if i == 0 and j == 1:
                y.append(1)
            if i == 1 and j == 0:
                y.append(2)
            if i == 1 and j == 1:
                y.append(3)
        y = pd.DataFrame({'labels':y})

        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(y)
        y = enc.transform(y).toarray().astype(int)
        y = pd.DataFrame(y)
        return y



    # private method
    def restaurant_codes(self):
        save_ids = []
        for idx, r in enumerate(self.review_data):

            if r['categories'] is not None and "restaurants" in r['categories'].lower():
                save_ids.append(r['business_id'])
        self.restaurant_ids = save_ids

    def find_restaurants(self, save=False):
        '''
        from the original dataset it only uses the reataurants review
        :param save: if we want to save the datasets
        :return: restaurants
        '''
        business_id = self.dataset['business_id'].values
        idx = np.array([])
        for i in range(len(self.restaurant_ids)):
            # print(len(restaurants_ids) - i)
            tmp = np.where(business_id == self.restaurant_ids[i])[0]
            idx = np.concatenate((idx, tmp))
        idx = idx.astype(int)  # these are the valid codes of the review we want

        restaurants = self.dataset.loc[idx, :][['text', 'stars']]
        restaurants = restaurants.reset_index()
        restaurants = restaurants.drop(['index'], axis=1)

        restaurants = self.filter_sentence(restaurants)
        self.restaurants = restaurants
        return self.restaurants

    def filter_sentence(self, dataset):
        '''

        :param dataset: is the dataset we want to filter by removing all sentences which have more than 5 sentences
        :param threshold: the # of sentences we do not want to exceed
        :return: filtered dataset
        '''
        idx = []
        for i in range(len(dataset)):
            if i % 100 == 0:
                print(len(dataset) - i)
            text = dataset['text'].iloc[i]
            text = sent_tokenize(text)
            size = len(text)
            if size < self.sent_thr:
                idx.append(i)
        dataset = dataset.iloc[idx, :]
        dataset = dataset.reset_index()
        dataset = dataset.drop(['index'], axis=1)
        dataset.to_csv("Cache/restaurants_" + str(self.sent_thr) + ".csv", index=False)
        return dataset

    def fix_word(self, word):

        fix_re = re.compile(r"[^a-z0-9.!,]+")
        num_re = re.compile(r'[0-9]+')
        word = word.lower()
        word = fix_re.sub('', word)
        word = num_re.sub('#', word)

        if not any((c.isalpha() or c in string.punctuation) for c in word):
            word = ''
        return word

    def language_cleaning(self, dataset):
        no_eng_idx = []
        for idx, data in enumerate(dataset.text):

            if len(data) <= 2 or detect(data) != 'en':
                no_eng_idx.append(idx)
        dataset = dataset.drop(no_eng_idx, axis=0)
        return dataset

    def preprocessing(self, dataset):
        reviews = []
        i = 0

        #dataset = self.language_cleaning(dataset)
        #dataset = dataset.reset_index().drop(['index'], axis = 1)

        tokenizer = PunktSentenceTokenizer()
        data = dataset['text']
        for idx, review in enumerate(data):
            collected_words = []
            if idx%1000==0:
                print(idx)
            #for sentence in tokenizer.tokenize(review):
            words = word_tokenize(review)
            if len(words) > self.word_thr:
                words = words[:self.word_thr]
            words = [self.fix_word(word) for word in words]
            #collected_words += words

            review = ' '.join(words)
            reviews.append(review)
            i += 1

        reviews = pd.DataFrame({'text': reviews})
        return reviews

    def dataset_report(self, dataset, attribute='sentiment'):

        global positive, negative, neutral
        if attribute == "sentiment":
            labels = dataset['stars'].values
            positive = sum(labels == 4) + sum(labels == 5)
            negative = sum(labels == 0) + sum(labels == 1) + sum(labels == 2)
            neutral = sum(labels == 3)
        print("Positive: {} | Negative: {} | Neutral: {} ".format(positive, negative, neutral))




datapath = "Yelp_data/yelp_review.csv"
samples = 10
data_review = "Yelp_data/yelp_academic_dataset_business.json"
n_classes = {'sentiment':2, 'tense':2}
threshold = {'sentence':10, 'word':15}
storing = {'path':"data/", 'test_size':0.1}
adapter = DatasetAdapter(datapath=datapath, samples=samples, data_review=data_review, threshold=threshold, n_classes=n_classes, assignment='hard', storing=storing)
#sentiment = pd.read_csv("sentimet_data/sentiment.csv")

#sentiment = sentiment.drop([60988], axis = 0).reset_index()
#sentiment = sentiment.dropna(axis=0)

X= adapter.adaptation()
#adapter.storing(X, y)




#sentiment = pd.read_csv("sentimet_data/sentiment.csv")
#sentiment_proc = pd.read_csv("sentimet_data/sentiment_proc.csv")



dataset = pd.read_csv(datapath)
