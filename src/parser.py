# -*- coding: utf-8 -*-
import xgboost as xgb

#TODO: implement correct analysis of imput sequence shift. Special attention to punctuation marks

import subprocess
import json
from gensim.models import KeyedVectors
from pymystem3 import Mystem
from nltk.tokenize import *
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
import math
# from sklearn import linear_model
from math import log

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import SGD


from sklearn import linear_model
import pickle
import os
import numpy as np


def decorate_file_name(f):
    return "../corpus/rucoref_texts/" + f + '_parsed'


punct_marks = ['...', '.', ',', '!', '?', ':', ';', '&', '%', '(', ')', '-', '--', '"']

pronoun_text_list = ["он", "его", "него", "ему", "нему", "им", "ним", "нем", "нём",
                     "она", "ее", "её", "нее", "неё", "ей", "ней", "ею", "нею",
                     "оно",
                     "они", "их", "них", "им", "ним", "ими", "ними"]

pronoun_feature_list = {}

pronoun_feature_list["он"] = "Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing"
pronoun_feature_list["него"] = "Animacy=Anim|Case=AccGen|Gender=MascNeut|Number=Sing"
pronoun_feature_list["его"] = "Animacy=Anim|Case=AccGen|Gender=MascNeut|Number=Sing"
pronoun_feature_list["ему"] = "Animacy=Anim|Case=Dat|Gender=MascNeut|Number=Sing"
pronoun_feature_list["нему"] = "Animacy=Anim|Case=Dat|Gender=MascNeut|Number=Sing"
pronoun_feature_list["им"] = "Animacy=Anim|Case=DatIns"
pronoun_feature_list["ним"] = "Animacy=Anim|Case=InsDat|Gender=MascNeut"
pronoun_feature_list["нем"] = "Animacy=Anim|Case=Loc|Gender=MascNeut|Number=Sing"
pronoun_feature_list["нём"] = pronoun_feature_list["нем"]

pronoun_feature_list["она"] = "Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing"
pronoun_feature_list["ее"] = "Animacy=Anim|Case=AccGen|Gender=Fem|Number=Sing"
pronoun_feature_list["её"] = pronoun_feature_list["ее"]
pronoun_feature_list["нее"] = "Animacy=Anim|Case=AccGen|Gender=Fem|Number=Sing"
pronoun_feature_list["неё"] = pronoun_feature_list["нее"]
pronoun_feature_list["ей"] = "Animacy=Anim|Case=DatIns|Gender=Fem|Number=Sing"
pronoun_feature_list["ею"] = "Animacy=Anim|Case=Ins|Gender=Fem|Number=Sing"
pronoun_feature_list["ней"] = "Animacy=Anim|Case=InsLoc|Gender=Fem|Number=Sing"
pronoun_feature_list["нею"] = "Animacy=Anim|Case=InsLoc|Gender=Fem|Number=Sing"

pronoun_feature_list["оно"] = "Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing"

pronoun_feature_list["они"] = "Case=Nom|Number=Plur"
pronoun_feature_list["их"] = "Case=AccGen|Number=Plur"
pronoun_feature_list["них"] = "Case=AccGenLoc|Number=Plur"
# pronoun_feature_list["им"] = "Case=DatIns"
# pronoun_feature_list["ним"] = "Case=Dat|Number=Plur"
pronoun_feature_list["ими"] = "Animacy=Anim|Case=Ins|Number=Plur"
pronoun_feature_list["ними"] = "Animacy=Anim|Case=Ins|Number=Plur"


pronoun_groups = []
pronoun_groups.append(pronoun_text_list[:9])
pronoun_groups.append(pronoun_text_list[9:18])
pronoun_groups.append(pronoun_text_list[19:])
pronoun_groups.append([pronoun_text_list[18]] + pronoun_text_list[1:9])


summ = 0
ans = 0


#TODO: put it into a new class called "features"
def decorate_features(feats):

    features_names =\
        "delta(id) \
        delta(sent) \
        delta(sh) \
        deprel(pos_feature) \
        case \
        animacy \
        parallelism \
        frequency \
        possible_coreference(assitiations_num) \
        number_of_pronoun \
        is_pronoun".split()
    return list(zip(feats, features_names))



# class MyNormalizer(object):
#     def __init__(self):
std = None
mean = None
if os.path.exists("stdmean"):
    with open("stdmean", 'rt') as f:
        std, mean = json.load(f)
        std = np.array(std)
        mean = np.array(mean)

def normalize( matrix):
    global std
    global mean
    if std is None and mean is None:
        std = np.std(matrix, axis=0)
        # print(std.reshape(1,2))
        mean = np.mean(matrix, axis=0)

    std_m = np.repeat(std.reshape(1, std.shape[0]), matrix.shape[0], 0)
    mean_m = np.repeat(mean.reshape(1, std.shape[0]), matrix.shape[0], 0)

    tmp = matrix - mean_m

    if not os.path.exists("stdmean"):
        with open("stdmean", 'wt') as f:
            json.dump((std.tolist(), mean.tolist()), f)
    # print("std = ", std)
    # print("mean = ", mean)

    return tmp / std_m


class AbstractWord:
    def __init__(self):
        pass

    def __parse_features__(self, morph_features):
        features = morph_features.split('|')
        parsed_features_tmp = [f.split('=') for f in features]
        parsed_features = {key:value for key, value in parsed_features_tmp}
        return parsed_features

    def get_feature(self, feature):
        return self.parsed_features.get(feature)

class Word(AbstractWord):
    def __init__(self, word_dict, sh):
        self.sh = sh
        self.field_dict = word_dict
        self.parsed_features = self.__parse_features__(self.field_dict['morph features'])
        self.antecedent_sh = -1      #for pronouns only

    def field(self, field):
        return self.field_dict[field]

class PronounInfo(AbstractWord):
    def __init__(self, text, features):
        self.text = text
        self.parsed_features = self.__parse_features__(features)

    def get_text(self):
        return self.text

class Sentence:
    def __init__(self, words_list, index):
        self.list = words_list
        self.index = index
        # print self.list

    def get_word_list(self):
        return self.list

    def find_word_by_id(self, word_id):
        tmp_list = [i for i in self.list if i.field('index') == word_id]
        if tmp_list:
            return tmp_list[0]
        else:
            return None
    def find_word(self, word_sh):
        tmp_list = [i for i in self.list if i.sh == word_sh]
        if tmp_list:
            return tmp_list[0]
        else:
            return None

    def find_in_sentence(self, sent_word_id):
        tmp_list = [i for i in self.list if i.field('index in sentence') == sent_word_id]
        if tmp_list:
            return tmp_list[0]
        else:
            return None

    def get_num_occ(self, word_text):
        i = 0;
        for word in self.get_word_list():
            if word_text == word.field('text'):
                i += 1
        return i


class Text:
    def __init__(self, sentences_list):
        self.sent_list = sentences_list

    def find_word(self, word_sh):
        for i, sent in enumerate(self.sent_list):
            tmp = sent.find_word(word_sh)
            if tmp is not None:
                # tmp.field_dict['sentence'] = i
                return tmp
        return None

    def find_word_by_id(self, word_id):
        for i, sent in enumerate(self.sent_list):
            tmp = sent.find_word_by_id(word_id)
            if tmp is not None:
                # tmp.field_dict['sentence'] = i
                return tmp
        return None

    def get_sentence(self, sent_num):
        tmp = [i for i in self.sent_list if i.index == sent_num]
        if len(tmp) == 1:
            return tmp[0]
        else:
            return None

    def get_sent_list(self):
        return self.sent_list

    def forward(self, sentence):
        self.sent_list = self.sent_list[1:]
        if sentence is not None:
            self.sent_list.append(sentence)

class TextBuilder:

    def __init__(self, file_name = None, doc_id = None):

        self.word_id = 0
        self.sent_id = 0

        self.doc_id = 0

        self.file_name = None
        self.text = None

        # print(file_name)

        self.word_sh = 0

        self.tokens_file = open("/home/gand/death/Diploma/corpus/Tokens.txt", 'rt')

        self.file = None

        # for excessive reading of the first word of the next sentence
        self.__tmp_first_word = None

        if file_name is not None:
            self.init_with_file(file_name, doc_id)

        return

    def init_with_file(self, file_name, doc_id):

        line = None

        if self.file is not None:
            self.file.close()
        self.file = open(file_name, 'rt')

        if decorate_file_name(file_name) != self.file_name and doc_id is not None:
            self.__tmp_first_word = None
            if self.file is not None:
                self.file.close()
            self.file_name = decorate_file_name(file_name)
            self.file = open(self.file_name, 'rt')

            if (doc_id < self.doc_id):
                # returning to the beginning of tokens file if tokens of newly opened file are above current position
                self.tokens_file.seek(0)

            line = self.tokens_file.readline()
            while (int(line.split()[0]) != doc_id):
                line = self.tokens_file.readline()
                    # print(line.split())

        self.doc_id = doc_id

        sentences = []

        sentences.append(self.read_sentence(line))
        while len(sentences) < 7:
            tmp = self.read_sentence()
            if tmp is not None:
                sentences.append(tmp)
            else:
                break
        self.text = Text(sentences)


    def read_sentence(self, first_line = None):

        tmp_sent = []
        if self.__tmp_first_word is not None:
            tmp_sent.append(self.__tmp_first_word)

        tokens_line = first_line

        while(True):
            line = self.file.readline()
            if line == '':
                if len(tmp_sent) > 1:
                    return Sentence(tmp_sent, self.sent_id)
                else:
                    return None

            spl = line.split()

            if len(spl) < 1:
                continue

            if tokens_line is None and self.doc_id is not None:
                tokens_line = self.tokens_file.readline()

            # tokens_line = self.tokens_file.readline()

            if tokens_line is not None:
                tok_spl = tokens_line.split()

            # print(tok_spl)

            # if we already passed first word of new text and read corresponding token string - we should pass it
            if self.doc_id is not None and self.__tmp_first_word is not None and tok_spl[1] == self.__tmp_first_word.sh:
                tokens_line = self.tokens_file.readline()
                tok_spl = tokens_line.split()

            if (len(spl) < 2):
                print("too short", spl, tok_spl)

            tmp_word = dict()
            tmp_word['string'] = spl
            tmp_word['index'] = self.word_id
            tmp_word['index in sentence'] = spl[0]
            tmp_word['text'] = spl[1].lower()
            tmp_word['postag'] = spl[3]
            tmp_word['punct text'] = spl[4]
            tmp_word['morph features'] = spl[5]
            tmp_word['head'] = spl[6]
            tmp_word['deprel'] = spl[7]
            tmp_word['sentence'] = self.sent_id

            if tokens_line is not None:
                self.word_sh = int(tok_spl[1])
            else:
                self.word_sh += len(tmp_word['text']) + 1

            # if tok_spl[3].lower() != tmp_word['text']:
            #     print("mismatch! ", self.word_sh, '"', tmp_word['text'], '"', tmp_word['string'], tok_spl[3].lower(), "doc", tok_spl[0])
            #     return None

            new_word = Word(tmp_word, self.word_sh)

            # self.word_sh += len(tmp_word['text'])
            # if(tmp_word['text'] not in punct_marks):
            #     # work is not punctiation mark - we need to add space
            #     self.word_sh += 1

            # tmp_word is a start of the new sentence
            if int(spl[0]) == 1:
                if self.word_id > 0:
                    # store tmp_word for the next sentence
                    self.__tmp_first_word = new_word
                    self.word_id += 1
                    new_sent = Sentence(tmp_sent, self.sent_id)
                    self.sent_id += 1
                    return new_sent

            tmp_sent.append(new_word)
            self.word_id += 1

            tokens_line = None

            # tokens_line = self.tokens_file.readline()



    def forward(self):
        sentence = self.read_sentence()
        self.text.forward(sentence)

    def get_text(self):
        return self.text




class Resolver:

    def __init__(self, paths, file_id, pronoun_list):

        # self.paths = paths
        # input_file_name = self.paths[file_id]
        #
        self.text_builder = TextBuilder()
        # self.file_id = file_id
        #
        # self.text = self.text_builder.get_text()

        self.resolved = dict()
        self.been_candidate = dict()

        self.pronoun_list = pronoun_list
        # self.way = way
        self.associations = dict()
        self.model = KeyedVectors.load_word2vec_format('/home/gand/lib/word2vec/ruscorpora.model.bin', binary=True)
        # self.stemmer = SnowballStemmer('russian')
        self.mystem = Mystem()

        self.s_dist_list = []

        self.answer_list = []
        self.answer_dict = dict()

        self.length = 10

        # self.classifier = linear_model.LogisticRegression(solver='liblinear')

        # self.classifier = RandomForestClassifier(max_features=5, max_depth= 7, n_estimators= 40, class_weight = {0:1, 1:9})
        # self.classifier = xgb.XGBClassifier(7, 0.05, 500)

        self.classifier = RandomForestClassifier(max_features="log2", n_estimators=6)

        # self.classifier = SVC(C=0.5, gamma=1/5., class_weight = {0:4 , 1:6}, probability=True)
        # self.classifier = xgb.XGBClassifier()


        # self.model = Sequential()
        # self.model.add(Dense(units=50, activation='relu', input_dim=11))
        # self.model.add(Dense(units=50, activation='tanh'))
        # self.model.add(Dense(units=30, activation='relu'))
        # self.model.add(Dense(units=20, activation='tanh'))
        # self.model.add(Dense(units=10, activation='relu'))
        # self.model.add(Dense(units=2, activation='softmax'))
        #
        # # opt = SGD(lr=0.05, momentum=0.0, nesterov=False)
        #
        # self.model.compile(optimizer='sgd', loss='mse')


        # self.scaler = MyNormalizer()

        self.coefficients = [-5, -7, 30, 10, 10, 10, 7, -50, 50]

        # self.classifier = linear_model.LogisticRegression(solver='liblinear', verbose=0)
        # self.classifier = svm.SVC(C = 10, probability=True)
        # self.classifier = tree.DecisionTreeClassifier(class_weight={0:1, 1:10}, max_depth=4)

    def import_answers(self, file):
        file = open(file)
        self.answer_list = []
        self.answer_dict = dict()
        for line in file:
            line = line.split()
            self.answer_list.append(int(line[1]))
            self.answer_dict[int(line[0])] = int(line[1])


    def get_training_examples_set(self, pronoun, text_id):
        candidates = self.build_candidates_list(pronoun)

        if pronoun.sh not in self.answer_dict[text_id]:
            print ("pronoun not in answer dict ", pronoun.field("text"), " ", pronoun.sh, ' text ', text_id)
            return None

        answer_sh = self.answer_dict[text_id][pronoun.sh]
        if len([c for c in candidates if c.sh == answer_sh]) == 0:
            cand_info = [(i.field('text'), i.sh) for i in candidates]
            print ("for pronoun ", pronoun.field('text'), " at ", pronoun.sh, " found no answer in candidates list: ", cand_info)
            pronoun.antecedent_sh = 0
            return None

        good_candidates = [c for c in candidates if c.sh > answer_sh and c.sh < pronoun.sh]

        global summ
        global ans

        summ += len(good_candidates)
        ans += 1

        examples_set = []
        for gc in good_candidates:
            features = self.build_features(gc, pronoun)
            examples_set.append((features, [1, 0], 0))

        answer = [c for c in candidates if c.sh == answer_sh][0]

        self.associations[pronoun] = answer

        answer_features = self.build_features(answer, pronoun)
        examples_set.append((answer_features, [0, 1], 1))

        return examples_set


    def fit(self, fit_paths):

        all_features_array = []
        all_answers_array_cls = []
        all_answers_array_nn = []

        if os.path.exists("classifier"):
            print("loading from file")
            with open("classifier", 'rb') as f:
                self.classifier = pickle.load(f)
                print(self.classifier.get_params())

            if os.path.exists("nn.h5"):
                self.model = load_model('nn.h5')

            return

        prons = 0
        for doc_id, path in fit_paths:
            print("document ", doc_id)
            self.text_builder.init_with_file(path, doc_id)
            self.text = self.text_builder.get_text()
            while len(self.text.get_sent_list()) > 0:
                self.build_prediction_list()
                for pronoun in self.pred_list:
                    prons += 1

                    examples = self.get_training_examples_set(pronoun, doc_id)
                    if examples is None:
                        continue

                    for ex in examples:
                        # print("feature = ", ex[0][0][8])
                        all_features_array.append(ex[0].reshape(-1,1))
                        all_answers_array_cls.append(ex[2])
                        all_answers_array_nn.append(ex[1])

                self.text_builder.forward()
                self.text = self.text_builder.get_text()
            print("prons in text: ", prons)

            self.associations = dict()
            self.been_candidate = dict()

        # print(len(all_features_array), len(all_answers_array), prons)
        # print(np.array(all_features_array).shape, np.array(all_answers_array).shape)

        # param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
        num_round = 2

        tmp_ar = np.array(all_features_array).reshape(len(all_features_array), len(all_features_array[0]))
        # print(tmp.shape, tmp[:10])

        # tmp = normalize(tmp_ar)

        tmp = tmp_ar
        # tmp = self.scaler.normalize(tmp)

        self.classifier.fit(tmp, np.array(all_answers_array_cls))

        print(self.classifier.get_params())

        l = len(all_features_array)
        # self.model.fit(tmp[:int(l * 0.8)], np.array(all_answers_array_nn[:int(l * 0.8)]), epochs=30, batch_size=10,
        #                class_weight=[1, 1], validation_data=(tmp[int(l * 0.8):], np.array(all_answers_array_nn[int(l * 0.8):])))

        # print("did fit", self.classifier.feature_importances_)

        with open("classifier", 'wb') as f:
            pickle.dump(self.classifier, f)

        # with open("nn.h5", 'wb') as f:
        # self.model.save("nn.h5")

        # with open("scaler", 'wb') as f:
        #     pickle.dump(self.scaler, f)
            # pickle.dump(s, f)

        # new_binary_set = classifier.predict(np.array(all_features_array))
        # print self.classifier.score(np.array(all_features_array), np.array(self.binary_set))

    def build_prediction_list(self):
        self.pred_list = []
        for sentence in self.text.get_sent_list():
            for word in sentence.get_word_list():
                if word.field('text').lower() in pronoun_text_list and word.antecedent_sh < 0:
                    self.pred_list.append(word)


    def evaluate(self, answers):
        n = 85
        fit_paths = list(self.paths.items())[n:]

        # # # #
        cls.answer_dict = answers

        cls.fit(fit_paths)

        # print(paths[list(paths.keys())[63]])

        tmp = []
        p_sum = 0
        r_sum = 0
        for path in list(paths.keys())[:n]:
        # for path in [12]:
            res = cls.predict_proba(path)

            p_sum += res[0]
            tmp.append(res[0])
            r_sum += res[1]
        #
        # print(list(zip(tmp, list(paths.keys())[:n])))

        precision = p_sum / n
        recall = r_sum/n
        f = 2 * precision * recall / (precision + recall)

        print("final", precision, recall, f)


    def predict_proba(self, doc_id, in_path = None):

        # print("classifier", decorate_features(self.classifier.feature_importances_))
        self.been_candidate = dict()
        self.associations = dict()

        global mean
        global  std

        # print("scaler", mean, std)
        if in_path != None:
            path = in_path
            doc_id = None
        else:
            path = self.paths[doc_id]

        # print(self.paths[doc_id])

        self.text_builder.init_with_file(path, doc_id)
        self.text = self.text_builder.get_text()

        if doc_id is None:
            doc_id = -1

        i = 0.0
        j = 0.0
        k = 0.0
        self.not_founders = 0
        while len(self.text.get_sent_list()) > 0:
            self.build_prediction_list()
            for pronoun in self.pred_list:
                k += 1.0
                res = self.predict_pronoun_proba(doc_id, pronoun)
                try:
                    # answer_sh = self.answer_dict[doc_id][pronoun.sh]
                    answer_sh = self.answer_dict[doc_id][pronoun.field('index')]

                except:
                    print ("not found pronoun at", pronoun.sh, "at answer dict of doc", doc_id)
                    continue

                if res is None or answer_sh is None:
                    continue
                j += 1

                if answer_sh == res:
                    i += 1.0
                else:
                    rw = self.text.find_word_by_id(res)
                    aw = self.text.find_word_by_id(answer_sh)
                    if rw is None:
                        print("wrong! ", res, "not found in text!", "instead of", answer_sh, aw.field("text"))
                    elif aw is None:
                        print("wrong! ", res, rw.field("text"), "instead of", answer_sh, "not found in text")
                    else:
                        print("wrong! ", res, rw.field("text"), "instead of", answer_sh, aw.field("text"))

            self.text_builder.forward()
            self.text = self.text_builder.get_text()


        print("i = {}, j = {}, k = {}".format(i,j,k))
        return (i/j, i/ k)

    def predict_pronoun_proba(self, text_id, pronoun):
        cand_list = self.build_candidates_list(pronoun)

        print("pronoun is ", pronoun.field('text'), pronoun.sh)

        if cand_list is None or len(cand_list) == 0:
            print ("no candidates for pronoun ", pronoun.field("text"), "at ", pronoun.sh)
            return None

        reverse_probas = dict()
        for candidate in cand_list:
            feats = self.build_features(candidate, pronoun)

            # nfeats = normalize(feats)
            nfeats = feats
            #
            res1 = self.classifier.predict_proba(feats)
            res = res1[0]

            # print("candidate", candidate.field("text"), candidate.sh, '-', res, '\n', decorate_features(nfeats.tolist()[0]), '\n')
            # reverse_probas[res[0][1]] = candidate

            reverse_probas[res[1] / nfeats[0][0]] = candidate

        antecedent = reverse_probas[max(reverse_probas.keys())]

        # print("pronoun ", pronoun.field('text'), "at ", pronoun.sh, "refers to ", antecedent.field('text'), "at ", antecedent.sh, '\n\n')
        print("pronoun ", pronoun.field('text'), "at ", pronoun.field('index'), "refers to ", antecedent.field('text'), "at ", antecedent.field('index'), '\n\n')


        # pronoun.antecedent_sh = antecedent.sh
        pronoun.antecedent_sh = antecedent.field('index')

        self.associations[pronoun] = antecedent

        return antecedent.field('index')
        # return antecedent.sh


    def build_candidates_list(self, pronoun, length = 10):
        # sent_num = pronoun.field('sentence')
        area = []
        txt = self.text.get_sent_list()
        for sent in txt:
            for word in sent.get_word_list():
                if pronoun.field("index") - word.field("index") < 35:
                    area.append(word)

        candidates = []

        for word in area:
            if self.is_word_acceptable(pronoun, word):
                candidates.append(word)
                tmp_word = self.mystem.lemmatize(word.field('text').lower())[0]
                if tmp_word in self.been_candidate:
                    self.been_candidate[tmp_word] += 1
                else:
                    self.been_candidate[tmp_word] = 1
        return candidates

    def predict(self, doc_id, in_path = None):
        i = 0.
        j = 0.

        k = 0.

        if in_path is None:
            path = self.paths[doc_id]
        else:
            path = in_path
            doc_id = None

        self.text_builder.init_with_file(path, doc_id)
        self.text = self.text_builder.get_text()

        while len(self.text.get_sent_list()) > 0:
            self.build_prediction_list()
            print(self.pred_list)
            for pronoun in self.pred_list:
                tmp = self.predict_word(pronoun.sh)
                k += 1
                if tmp is not None:
                    j += 1
                    # print(pronoun.field('text'), pronoun.sh, "refers to", tmp.field('text'), tmp.sh)
                    print(pronoun.field('text'), pronoun.field('index'), "refers to", tmp.field('text'), tmp.field('index'))

                    try:
                        if self.answer_dict[doc_id][pronoun.field('index')] == tmp.field('index'):
                            i += 1
                        else:
                            print("wrong! ", tmp.field('index'), "instead of", self.answer_dict[doc_id][pronoun.field('index')])
                    except:
                        print("not found in answer list", doc_id, pronoun.field('index'))
            self.text_builder.forward()
            self.text = self.text_builder.get_text()

        # print ("j and deb", j, deb)
        print("p, r, f", i / j, i / k)
        return (i, j, k)

    def predict_word(self, num_pron):
        pronoun = self.text.find_word(num_pron)

        if pronoun is None:
            print("NONe!", num_pron)

        # sent_num = pronoun.field('sentence')

        candidates = self.build_candidates_list(pronoun, self.length)

        if len(candidates) == 0:
            print("error! no candidates", pronoun.field('text'), pronoun.field('index'))
            return None

        self.features = {}
        for candidate in candidates:
            self.features[candidate] = self.build_features(candidate, pronoun)

        antecedent = self.get_right_word(pronoun)
        self.associations[pronoun] = antecedent

        pronoun.field('string')[9] = "refto_" + str(antecedent.sh)

        pronoun.antecedent_sh = antecedent.sh

        self.features = dict()

        return antecedent
            # print candidate.field('text'), self.features[candidate]

    def get_right_word(self, pronoun):
        score_list = []
        for candidate in self.features.keys():

            tmp_features = [a * b for (a, b) in zip(self.features[candidate], self.coefficients)]

            score = sum(tmp_features[0])
            score_list.append(score)


        pass
        res = list(self.features.keys())[score_list.index(max(score_list))]

        self.s_dist_list.append(self.features[res][0])

        # print self.features[res], "- features of candidate, score =", max(score_list)

        try:
            ans_feat = self.features[self.text.find_word(self.answer_list[self.pred_list.index(pronoun)])]
            # print ans_feat, "features of answer, score = ", sum([a * b for (a, b) in zip(ans_feat, self.coefficients)])
        except:
            if len(self.answer_list) > 0:
                print("not found in candidate list!", self.answer_list[self.pred_list.index(pronoun)])
            # self.not_founders += 1

        return res

    def is_there_coreference(self, left_pron, right_pron):

        if left_pron.field('head') == right_pron.field('head') and \
                left_pron.field('sentence') == right_pron.field('sentence'):
            return False

        set1 = ["он", "его", "него", "ему", "нему", "им", "ним", "нем", "нём"]
        set2 =  ["она", "ее", "её", "нее", "неё", "ей", "ней", "ею", "нею"]

        if left_pron.field("sentence") == right_pron.field('sentence'):
            if left_pron.field('text') in set1 and right_pron.field('text') in set1 or \
                    left_pron.field('text') in set2 and right_pron.field('text') in set2:
                return True

        if left_pron.field("sentence") == right_pron.field('sentence') - 1:
            if left_pron.field('text') == right_pron.field('text'):
                return True
        else:
            return False

    def get_word_frequency(self, word_text):
        try:
            return self.been_candidate[word_text]
        except KeyError:
            print("key error at get frequency", self.been_candidate)


    def build_features(self, candidate, pronoun):
        res = []
        if candidate.field("postag") == 'PRON':
            res =  self.build_pronoun_features(candidate, pronoun)
        else:
            res =  self.build_features_list(candidate, pronoun)
        # print("len", len(res))
        npres = np.array(res).reshape(1,-1)
        # return normalize(npres)
        return npres

    def build_features_list(self, candidate, pronoun):

        pronoun_info = [i for i in self.pronoun_list if i.get_text() == pronoun.field('text')][0]
        features_list = []

        # features[0]
        # distance between candidate and pronoun. Might be negative, if candidate is further then the pronoun
        delta = pronoun.field('index') - candidate.field('index')
        # if delta < 0:
        #     delta = 10000

        delta **= 2
        features_list.append(delta)

        # features[1]
        #number of sentences between candidate and pronoun. Also might be negative
        #
        # sent_delta = pronoun.field('sentence') - candidate.field('sentence')
        #
        # if sent_delta < 0:
        #     sent_delta = 100
        # features_list.append(sent_delta)


        sh_delta = pronoun.sh - candidate.sh
        # if (sh_delta < 0):
        #     print("stang, sh_delta < 0", sh_delta)
        #     sh_delta = 1000000
        features_list.append(sh_delta)

        # features[2]
        #feature connected with candidates position in a sentence
        pos_feature1 = 0
        pos_feature2 = 0

        if candidate.field('deprel') == 'nsubj' or candidate.field('postag') == 'nsubjpass':
            pos_feature1 = 1
        elif candidate.field('deprel') == 'dobj':
            pos_feature2 = 1

        features_list.append(pos_feature1)
        features_list.append(pos_feature2)


        # :TODO frequency feature might be implemented
        # :TODO add word2vec feature if there will be any time left

        # features[3]
        # feature connected with the case of pronoun and it's antecedent
        # It's written like "in" here, because each pronoun may have several  variants of case
        case_feature = 0
        if candidate.get_feature('Case') is not None:
            if candidate.get_feature('Case') in pronoun_info.get_feature('Case'):
                case_feature = 1
        features_list.append(case_feature)


        # features[4]
        #feature connected with animacy of candidate
        animacy_feature = 0
        if pronoun_info.get_feature('Animacy') is not None and candidate.get_feature('Animacy') is not None:
            try:
                if candidate.get_feature('Animacy') in pronoun_info.get_feature('Animacy'):
                    animacy_feature = 1
            except:
                pass
        features_list.append(animacy_feature)


        # features[5]
        #simple parallelism feature
        synt_parallel_feature1 = 0
        synt_parallel_feature2 = 0
        synt_parallel_feature3 = 0
        if candidate.field("deprel") == pronoun.field("deprel"):
            synt_parallel_feature1 = 1
            # print("look for pronoun", pronoun.field('text'), pronoun.field('index') , "in sentence", pronoun.field('sentence') )
            try:
                head_pron = self.text.get_sentence(pronoun.field('sentence')).find_in_sentence(pronoun.field('head'))
                head_candidate = self.text.get_sentence(candidate.field('sentence')).find_in_sentence(candidate.field('head'))
                if head_pron.field('deprel') == 'ROOT':
                    synt_parallel_feature2 = 1
                if head_candidate is not None and head_pron is not None and head_candidate.field('deprel') == head_pron.field('deprel'):
                    synt_parallel_feature3 = 1
            except AttributeError as err:
                print("pronoun number", pronoun.field('sentence'), err.args)
                for i in [a.index for a in self.text.get_sent_list()]:
                    print("sentences:", i)

        features_list.append(synt_parallel_feature1)
        features_list.append(synt_parallel_feature2)
        features_list.append(synt_parallel_feature3)



        # features[6]
        #Frequency feature
        # frequency_feature = word.been_ante
        frequency_feature = self.get_word_frequency(self.mystem.lemmatize(candidate.field("text"))[0])
        features_list.append(frequency_feature)



        features_list.append(pronoun_text_list.index(pronoun.field('text')))


        # features[9]
        # appending zero feature only for noun candidates. It is set to one for pronouns
        features_list.append(0)

        #
        # print(candidate.field("text"))
        # print(features_list, '\n')


        return features_list

    def is_word_acceptable(self, pron, candidate):

        condition_list = []

        if candidate.sh > pron.sh:
            return False

        # leave nouns and pronouns

        if candidate.field('postag') not in  ['NOUN', 'PRON']:
            return False

        # print(pron.field('text'))
        pronoun_info = [i for i in self.pronoun_list if i.get_text() == pron.field('text')][0]


        if candidate.field('postag') == 'PRON':
            tmp = [i for i in pronoun_groups if candidate.field('text') in i and pron.field('text') in i]
            if len(tmp) == 0:
                # print ("bad pronoun ", candidate.field('text'), pron.field('text'))
                return False

        # pronoun and antecedent should be the same number
        if candidate.get_feature('Number') and pronoun_info.get_feature('Number') is not None:
            condition_list.append(candidate.get_feature('Number') in pronoun_info.get_feature('Number'))

        # need to do something with that. In syntaxnet gender doesn't always work correctly
        if candidate.get_feature('Gender')and pronoun_info.get_feature('Gender') is not None:
            condition_list.append(candidate.get_feature('Gender') in pronoun_info.get_feature('Gender'))

        fnd_sentence = self.text.get_sentence(candidate.field('sentence'))
        if fnd_sentence is None:
            # print("sentence not found!", candidate.field('sentence'))
            condition_list.append(0)
        else:
            tmp_word = fnd_sentence.find_in_sentence(candidate.field('head'))

            # arguments dependancy
            arg_dep = pron.field('head') != candidate.field('head') or \
                        pron.field('sentence') != candidate.field('sentence')

            condition_list.append(arg_dep)

        for i, cond in enumerate(condition_list):
            if not cond:
                return False
        return True


    def build_pronoun_features(self, candidate, pronoun):
        pronoun_info = [i for i in self.pronoun_list if i.get_text() == pronoun.field('text')][0]
        features_list = []
        if (pronoun.antecedent_sh > 0):
            # case where antecedent might be a pronoun, which had been processed previousle (coreference)
            # then we get the features from original NP with some changes
            # print("possible coreference ", pronoun.field("text"), pronoun.sh, "with", candidate.field("text"), candidate.sh)
            prev_cand = self.associations[candidate]
            if prev_cand is not None:
                features_list = self.build_features_list(prev_cand, pronoun)
                # features_list = self.build_features_list(pronoun_info, pronoun)

            if candidate.field('index') in self.associations.keys():
                tmp = self.associations[candidate]
                # delta = pronoun.field('index') - tmp.field('index')
                delta = pronoun.field('index') - candidate.field('index')
            else:
                delta = pronoun.field('index') - candidate.field('index')

            if delta < 0:
                delta = 100000
            features_list[0] = delta

            #number of sentences between candidate and pronoun. Also might be negative
            sent_delta = 0
            if candidate.field('sentence') in self.associations.keys():
                tmp = self.associations[candidate]
                # sent_delta = pronoun.field('sentence') - tmp.field('sentence')
                sent_delta = pronoun.field('sentence') - candidate.field('sentence')
            else:
                sent_delta = pronoun.field('sentence') - candidate.field('sentence')

            if sent_delta < 0:
                sent_delta = 10000
            features_list[1] = sent_delta
            features_list[-1] = 1
        else:
            candidate.parsed_features = pronoun_info.parsed_features
            features_list = self.build_features_list(candidate, pronoun)
            features_list[-1] = 1

        return features_list






#:TODO distribute the code to different files


# import pymorphy2
# morph = pymorphy2.MorphAnalyzer()

def basic_main():
    pronoun_list = [PronounInfo(i, pronoun_feature_list[i]) for i in pronoun_text_list]

    cls = Resolver("../output1.txt", pronoun_list)
    cls.import_answers("../answers")

    output_file = open("../result1", 'w')

    cls.import_answers("../answers")
    a1 = cls.predict()

    cls = Resolver("../output2.txt", pronoun_list)

    cls.import_answers("../answers2")
    a2 = cls.predict()

    print("purity = ", (a1[0] + a2[0]) / (a1[1] + a2[1]))



class TrainSampleData:

    def __init__(self, filename = None):
        self.doc_paths = dict()
        self.answers = dict()
        self.deps = dict()

        if filename is not None:
            self.parse_xml(filename)

    def load(self, filename):
        self.answers = dict()
        with open(filename, 'rt') as fl:
            while True:
                tmp = fl.readline().split()
                if len(tmp) == 0:
                    break
                print(tmp)
                self.answers[int(tmp[0])] = int(tmp[1])


    def parse_xml(self, filename):
        import xml.etree.ElementTree as ET
        tree = ET.parse(filename)
        root = tree.getroot()

        documents = root.findall(".//*[@doc_id]")
        for doc in documents:
            id = int(doc.get("doc_id"))
            self.doc_paths[id] = doc.get("path")

            # anaph_groups = doc.findall(".//attributes/*[@val='anaph']../..")
            anaph_groups = doc.findall(".//attributes/*[@val]../..")
            self.deps[id] = dict()

            for group in anaph_groups:
                if "link" in group.attrib:
                    link_sh = group.attrib['link']
                    # print("link: ", link_sh)
                    source = doc.find(".//*[@group_id='{}']".format(link_sh))

                    # print(source.get('sh'))
                    self.deps[id][group] = source

        for doc_id, doc_deps in self.deps.items():
            self.answers[int(doc_id)] = dict()
            for ref, src in doc_deps.items():
                tmp_src = src.find("./items")
                tmp_src_sh = 0

                # tmp_src_sh = int(tmp_src.find("./*").get('sh'))
                # tmp_src_len = int(tmp_src.find("./*").get('len'))

                # sequence of multiple words as antecedent NP
                if (len(list(tmp_src)) > 1):
                    # print(next(iter(tmp_src)).get('sh'))
                    head_item = tmp_src.find(".//*[@head='1']")
                    if head_item is not None:
                        tmp_src_sh = int(head_item.get('sh'))
                    else:
                        # special case - no head in noun phrase. Assume it consists of main word with punct. marks
                        head_item = [i for i in tmp_src.findall(".//*[@gram]") if i.get('gram') != '-'][0]
                        tmp_src_sh = int(head_item.get('sh'))
                else:
                    tmp_src_sh = int(tmp_src.find("./*").get('sh'))

                tmp_ref = ref.find("./items")
                tmp_ref_sh = 0
                if (len(list(tmp_ref)) > 1):
                    head_item = tmp_src.find(".//*[@head='1']")
                    if head_item is not None:
                        tmp_ref_sh = int(head_item.get('sh'))
                    else:
                        try:
                            head_item = [i for i in tmp_src.findall(".//*[@gram]") if i.get('gram') != '-'][0]
                        except:
                            head_item = [i for i in tmp_src.findall(".//*[@gram]")][0]
                            print("warning! bad head", [i.items() for i in tmp_src.findall(".//*[@gram]")])

                        tmp_ref_sh = int(head_item.get('sh'))
                else:
                    tmp_ref_sh = int(tmp_ref.find("./*").get('sh'))

                self.answers[int(doc_id)][tmp_ref_sh] = tmp_src_sh


    # print(list(answers[1].items())[:50])

    # print(len(paths), len(deps))


    # for conn in list(connections.items())[:10]:
    #     print(conn[0].get("link"), conn[1].get("group_id"))


        # print(doc.tag)

    # for attribute in root.iter('attributes'):
    #     print(attribute.find(''))

    # for document in root:
    #     chains = next(iter(document))
    #     for chain in chains:
    #         for group in chain:
    #             print(group.tag, group.attrib)

    # print(ana)


class file_parser:
    def __init__(self, dir, token_filename):
        self.dir = dir
        self.token_filename = token_filename
        self.toker = TweetTokenizer()
        # self.filename = filename

    def prepare_file(self, doc_id):
        # f = open(self.dir + filename, 'r')

        out = open(self.dir + 'tmp', 'w')

        token_file = open(self.token_filename, 'r')

        tokens = [i.split()[3] for i in token_file if int(i.split()[0]) == doc_id]

        # print("len = ", len(tokens))

        tmp_line = " ".join(tokens)
        tmp_line = tmp_line.replace('.', '.\n')
        tmp_line = tmp_line.replace(".\n.\n.\n", "...")

        out.write(tmp_line)

    # def prepare_file(self, filename):
    #     f = open(self.dir + filename, 'r')
    #     out = open(self.dir + 'tmp', 'w')
    #     for line in f:
    #         tmp_line = line
    #         for sym in punct_marks:
    #             if (sym in tmp_line):
    #                 tmp_line = tmp_line.replace(sym, ' ' + sym + ' ')
    #         #correct treatment for suspension points
    #         tmp_line = tmp_line.replace('.', '.\n')
    #         tmp_line = tmp_line.replace(".\n  .\n  .\n", "...")
    #
    #         out.write(tmp_line)
    #         # tmp_tokens = self.toker.tokenize(tmp_line)
    #         # out.write(' '.join(tmp_tokens))

    def parse_file(self, in_filename, out_filename):
        p = subprocess.call(["./src/scripts/parse_file.sh", self.dir, in_filename, out_filename])


    def parse_all_files(self, paths):
        for i, file in enumerate(list(paths.keys())):
            self.prepare_file(file)
            self.parse_file('tmp', paths[file] + "_parsed")
            print("parsed file", i)


# sample = TrainSampleData('../corpus/groups.xml')
#

sample1 = TrainSampleData()
sample1.load('a1')

# print(list(sample.doc_paths.items())[:10])
paths = sample1.doc_paths

pronoun_list = [PronounInfo(i, pronoun_feature_list[i]) for i in pronoun_text_list]


# cls = Resolver(sample.doc_paths, 1, pronoun_list)
cls = Resolver(None, 1, pronoun_list)
cls.answer_dict[-1] = sample1.answers

with open("classifier", 'rb') as f:
    cls.classifier = pickle.load(f)

i1, j1 = cls.predict_proba(None, "output1.txt")

sample = TrainSampleData()
sample.load('a2')
cls = Resolver(None, 1, pronoun_list)

with open("classifier", 'rb') as f:
    cls.classifier = pickle.load(f)

cls.answer_dict[-1] = sample.answers
i2, j2 = cls.predict_proba(None, "output2.txt")

print((i1 + i2) / 2, (j1 + j2) / 2)

