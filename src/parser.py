# -*- coding: utf-8 -*-
import xgboost as xgb

#TODO: implement correct analysis of imput sequence shift. Special attention to punctuation marks

import subprocess
import json
from gensim.models import KeyedVectors
from pymystem3 import Mystem
from nltk.tokenize import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
# from sklearn import linear_model
from math import log

from sklearn import linear_model
import pickle
import os
import numpy as np


# f = open("./parsed/output1.txt")
# file = f.read()
# f = file.split('\n')


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

#TODO: put it into a new class called "features"
def decorate_features(feats):

    features_names =\
        "delta(id) \
        delta(sent) \
        deprel(pos_feature) \
        case \
        animacy \
        parallelism \
        frequency \
        word2vec \
        possible \
        coreference(assitiations_num) \
        is_pronoun".split()
    return list(zip(feats, features_names))


std = None
mean = None
if os.path.exists("stdmean"):
    with open("stdmean", 'rt') as f:
        std, mean = json.load(f)
        std = np.array(std)
        mean = np.array(mean)
def normalize(matrix):
    global std
    global mean
    if std is None and mean is None:
        std = np.std(matrix, axis=0)
        # print(std.reshape(1,2))
        mean = np.mean(matrix, axis=0)

    std_m = np.repeat(std.reshape(1, std.shape[0]), matrix.shape[0], 0)
    mean_m = np.repeat(mean.reshape(1, std.shape[0]), matrix.shape[0], 0)

    tmp = matrix - mean_m

    # print("stdm" ,decorate_features(list(std)))

    if not os.path.exists("stdmean"):
        with open("stdmean", 'wt') as f:
            json.dump((std.tolist(), mean.tolist()), f)

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

    # def get_word_frequency(self, word_text):
    #     return self.been_candidate[word_text]
        # return sum([sent.get_num_occ(word_text) for sent in self.get_sent_list()])

    def forward(self, sentence):
        self.sent_list = self.sent_list[1:]
        if sentence is not None:
            self.sent_list.append(sentence)

class TextBuilder:

    def __init__(self, file_name, doc_id):

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

        self.init_with_file(file_name, doc_id)

        return

    def init_with_file(self, file_name, doc_id):

        line = None

        if decorate_file_name(file_name) != self.file_name:
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

        print("first line is", line)

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
        if self.__tmp_first_word != None:
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

            if tokens_line is None:
                tokens_line = self.tokens_file.readline()

            # tokens_line = self.tokens_file.readline()

            tok_spl = tokens_line.split()

            # print(tok_spl)

            # if we already passed first word of new text and read corresponding token string - we should pass it
            if self.__tmp_first_word is not None and tok_spl[1] == self.__tmp_first_word.sh:
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

            self.word_sh = int(tok_spl[1])

            if tok_spl[3].lower() != tmp_word['text']:
                print("mismatch! ", self.word_sh, '"', tmp_word['text'], '"', tmp_word['string'], tok_spl[3].lower(), "doc", tok_spl[0])
                return None

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

        self.paths = paths
        input_file_name = self.paths[file_id]

        self.text_builder = TextBuilder(input_file_name, file_id)
        self.file_id = file_id

        self.text = self.text_builder.get_text()

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

        self.classifier = RandomForestClassifier()
        # self.classifier = xgb.XGBClassifier()

        self.scaler = Normalizer()

        # purity = 52.2%

        # self.coefficients = [-0.05,-1,3,2,2,2,5,3]


        # purity = 54.7%
        # SET ONE
        # self.coefficients = [-10, -10, 10, 10, 10, 10, 10, 10, 10]

        # SET TWO
        # purity = 60%
        # self.coefficients = [-5, 0, 30, 20, 20, 20, 50, 70, 50]

        # purity > 62%
        # purity = 65.2% with area = 10
        # SET THREE

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
        examples_set = []
        for gc in good_candidates:
            features = self.build_features(gc, pronoun)
            examples_set.append((features, 0))

        answer = [c for c in candidates if c.sh == answer_sh][0]
        answer_features = self.build_features(answer, pronoun)
        examples_set.append((answer_features, 1))

        return examples_set


    def fit(self, fit_paths):

        all_features_array = []
        all_answers_array = []

        if os.path.exists("classifier"):
            print("loading from file")
            with open("classifier", 'rb') as f:
                self.classifier = pickle.load(f)

            with open("scaler", 'rb') as f:
                self.scaler = pickle.load(f)

            return



        for doc_id, path in fit_paths:
            print("document ", doc_id)
            prons = 0
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
                        all_features_array.append(ex[0].reshape(-1,1))
                        all_answers_array.append(ex[1])

                self.text_builder.forward()
                self.text = self.text_builder.get_text()
            print("prons in text: ", prons)

        # with open("training_data.json", 'w') as f:
        #     json.dump(all_features_array, f)
        #     json.dump(all_answers_array, f)

            print("data dumped")

        print(len(all_features_array), len(all_answers_array), prons)
        print(np.array(all_features_array).shape, np.array(all_answers_array).shape)

        param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
        num_round = 2

        # tmp = self.scaler.fit_transform(np.array(all_features_array).reshape(len(all_features_array[0]), len(all_features_array)))
        # tmp = tmp.reshape(len(all_features_array), len(all_features_array[0]))
        # tmp = np.array(all_features_array).reshape(len(all_features_array), len(all_features_array[0]))

        tmp = np.array(all_features_array).reshape(len(all_features_array), len(all_features_array[0]))
        print(tmp.shape, tmp[:10])

        tmp = normalize(tmp)

        self.classifier.fit(tmp, np.array(all_answers_array))

        print("did fit", self.classifier.feature_importances_)

        with open("classifier", 'wb') as f:
            pickle.dump(self.classifier, f)

        with open("scaler", 'wb') as f:
            pickle.dump(self.scaler, f)
            # pickle.dump(s, f)

        # new_binary_set = classifier.predict(np.array(all_features_array))
        # print self.classifier.score(np.array(all_features_array), np.array(self.binary_set))

    def build_prediction_list(self):
        self.pred_list = []
        for sentence in self.text.get_sent_list():
            for word in sentence.get_word_list():
                if word.field('text').lower() in pronoun_text_list and word.antecedent_sh < 0:
                    self.pred_list.append(word)


    def predict_proba(self, doc_id):

        print("classifier", decorate_features(self.classifier.feature_importances_))

        global mean
        global  std

        print("scaler", mean, std)

        path = self.paths[doc_id]

        print(self.paths[doc_id])

        self.text_builder.init_with_file(path, doc_id)
        self.text = self.text_builder.get_text()

        i = 0.0
        j = 0.0
        self.not_founders = 0
        while len(self.text.get_sent_list()) > 0:
            self.build_prediction_list()
            for pronoun in self.pred_list:
                j += 1
                res = self.predict_pronoun_proba(doc_id, pronoun)
                try:
                    answer_sh = self.answer_dict[doc_id][pronoun.sh]
                except:
                    print ("not found pronoun at", pronoun.sh, "at answer dict of doc", doc_id)
                    continue

                if res is None or answer_sh is None:
                    continue

                if answer_sh == res:
                    i += 1.0
                else:
                    rw = self.text.find_word(res)
                    aw = self.text.find_word(answer_sh)
                    if rw is None:
                        print("wrong! ", res, "not found in text!", "instead of", answer_sh, aw.field("text"))
                    elif aw is None:
                        print("wrong! ", res, rw.field("text"), "instead of", answer_sh, "not found in text")
                    else:
                        print("wrong! ", res, rw.field("text"), "instead of", answer_sh, aw.field("text"))

                # cand_list = self.build_candidates_list(pronoun.sh)
                # if cand_list is None:
                #     continue
                #
                # probas = dict()
                # for candidate in cand_list:


                # print(pronoun.field('text'), pronoun.sh, "refers to", tmp.field('text'), tmp.sh)
                # if len(self.answer_dict) > 0 and pronoun.sh in self.answer_dict[self.file_id]:
                #     if self.answer_dict[file_id][pronoun.sh] == tmp.sh:
                #         i += 1.0
                #     else:
                #         print("wrong! ", tmp.sh, "instead of", self.answer_dict[self.file_id][pronoun.sh])
            self.text_builder.forward()
            self.text = self.text_builder.get_text()

            # for s in self.text.get_sent_list():
            #     print(s.index)
            # print ("after_up_end\n")

        print("per cent = ", i/j)
        return (i/j, len(self.answer_list))


    def predict_pronoun_proba(self, text_id, pronoun):
        cand_list = self.build_candidates_list(pronoun)

        # print("pronoun is ", pronoun.field('text'))

        if cand_list is None or len(cand_list) == 0:
            print ("no candidates for pronoun ", pronoun.field("text"), "at ", pronoun.sh)
            return None

        reverse_probas = dict()
        for candidate in cand_list:
            feats = self.build_features(candidate, pronoun)
            # nfeats = self.scaler.transform(feats)

            nfeats = normalize(feats)

            # nfeats = normalize(feats)


            # dtest = xgb.DMatrix(feats)
            # res = self.classifier.predict_proba(feats)
            res = self.classifier.predict_proba(nfeats)

            print("candidate", candidate.field("text"), candidate.sh, res, '\n', decorate_features(feats.tolist()[0]), decorate_features(nfeats.tolist()[0]), '\n')
            reverse_probas[res[0][1]] = candidate

        antecedent = reverse_probas[max(reverse_probas.keys())]

        print("pronoun ", pronoun.field('text'), "at ", pronoun.sh, "refers to ", antecedent.field('text'), "at ", antecedent.sh, '\n\n')

        pronoun.antecedent_sh = antecedent.sh

        return antecedent.sh

    def build_candidates_list(self, pronoun, length = 10):
        sent_num = pronoun.field('sentence')
        area = self.text.get_sent_list()

        candidates = []
        for sentence in area:
            tmp = sentence.get_word_list()
            for word in tmp:
                if self.is_word_acceptable(pronoun, word):
                    candidates.append(word)
                    tmp_word = self.mystem.lemmatize(word.field('text').lower())[0]
                    if tmp_word in self.been_candidate:
                        self.been_candidate[tmp_word] += 1
                    else:
                        self.been_candidate[tmp_word] = 1
        return candidates

    def predict_word(self, num_pron):
        # word = self.text.find_word(num_word)
        pronoun = self.text.find_word(num_pron)

        sent_num = pronoun.field('sentence')

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
            if candidate.field('index') in self.associations.values():    #coreference!
                tmp_distances = []
                tmp_s_distances = []
                tmp = [i for i in self.associations.keys() if self.associations[i] == candidate]
                for pron2 in tmp:
                    if self.is_there_coreference(pronoun, pron2):
                        return candidate
                    tmp_distances.append(pronoun.field('index') - pron2.field('index'))
                    tmp_s_distances.append(pronoun.field('sentence index') - pron2.field('sentence index'))

                new_distance = min(tmp_distances)
                new_s_distance = min(tmp_s_distances)
                self.features[candidate][0] = new_distance
                self.features[candidate][1] = new_s_distance

            tmp_features = [a * b for (a, b) in zip(self.features[candidate], self.coefficients)]

            score = sum(tmp_features)
            score_list.append(score)

        res = list(self.features.keys())[score_list.index(max(score_list))]

        self.s_dist_list.append(self.features[res][1])

        # print self.features[res], "- features of candidate, score =", max(score_list)

        try:
            ans_feat = self.features[self.text.find_word(self.answer_list[self.pred_list.index(pronoun)])]
            # print ans_feat, "features of answer, score = ", sum([a * b for (a, b) in zip(ans_feat, self.coefficients)])
        except:
            if len(self.answer_list) > 0:
                print("not found in candidate list!", self.answer_list[self.pred_list.index(pronoun)])
            self.not_founders += 1


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
        if delta < 0:
            delta = 10000
        features_list.append(delta)

        # features[1]
        #number of sentences between candidate and pronoun. Also might be negative
        sent_delta = 0
        # if candidate.field('sentence') in self.associations.keys():
        #     tmp = self.associations[candidate.field('sentence')]
        #     sent_delta = pronoun.field('sentence') - tmp.field('sentence')
        # else:

        sent_delta = pronoun.field('sentence') - candidate.field('sentence')

        if sent_delta < 0:
            sent_delta = 100
        features_list.append(sent_delta)


        sh_delta = pronoun.sh - candidate.sh
        if (sh_delta < 0):
            print("stang, sh_delta < 0", sh_delta)
            sh_delta = 1000000
        features_list.append(sh_delta)

        # features[2]
        #feature connected with candidates position in a sentence
        pos_feature = 0
        if candidate.field('deprel') == 'nsubj' or candidate.field('postag') == 'nsubjpass':
            pos_feature = 2
        elif candidate.field('deprel') == 'dobj':
            pos_feature = 1

        features_list.append(pos_feature)

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
        synt_parallel_feature = 0
        if candidate.field("deprel") == pronoun.field("deprel"):
            synt_parallel_feature = 1
            # print("look for pronoun", pronoun.field('text'), pronoun.field('index') , "in sentence", pronoun.field('sentence') )
            try:
                head_pron = self.text.get_sentence(pronoun.field('sentence')).find_in_sentence(pronoun.field('head'))
            except AttributeError as err:
                print("pronoun number", pronoun.field('sentence'), err.args)
                for i in [a.index for a in self.text.get_sent_list()]:
                    print("sentences:", i)

            head_candidate = self.text.get_sentence(candidate.field('sentence')).find_in_sentence(candidate.field('head'))
            if head_candidate is not None and head_pron is not None and head_candidate.field('deprel') == head_pron.field('deprel'):
                synt_parallel_feature = 3
                if head_pron.field('deprel') == 'ROOT':
                    synt_parallel_feature = 6
        features_list.append(synt_parallel_feature)

        # features[6]
        #Frequency feature
        # frequency_feature = word.been_ante
        frequency_feature = self.get_word_frequency(self.mystem.lemmatize(candidate.field("text"))[0])
        features_list.append(frequency_feature)

        # # features[7]
        # #Using Word2Vec
        # similarity_feature = 0
        # left_neighbour = self.text.find_word_by_id(pronoun.field('index') - 1)
        #
        # if left_neighbour is not None and left_neighbour.field('punct text') != '_':
        #     left_neighbour = self.text.find_word_by_id(pronoun.field('index') - 2)
        #
        # right_neighbour = self.text.find_word_by_id(pronoun.field('index') + 1)
        # if right_neighbour is not None and right_neighbour.field('punct text') != '_':
        #     right_neighbour = self.text.find_word_by_id(pronoun.field('index') + 2)
        #
        # try:
        #     left_neighbour_lemma = self.mystem.lemmatize(left_neighbour.field('text'))[0]
        #     right_neighbour_lemma = self.mystem.lemmatize(right_neighbour.field('text'))[0]
        #
        #     candidate_lemma = self.mystem.lemmatize(candidate.field('text'))[0]
        #
        #     if left_neighbour_lemma is not None:
        #         similarity_feature += self.model.similarity(candidate_lemma.decode('utf8'),
        #                                                     left_neighbour_lemma.decode('utf8'))
        #     if right_neighbour_lemma is not None:
        #         similarity_feature += self.model.similarity(candidate_lemma.decode('utf8'),
        #                                                     right_neighbour_lemma.decode('utf8'))
        # except:
        #     pass
        #
        # features_list.append(similarity_feature)

        # # features[8
        # tmp = [i for i in self.associations.keys() if self.associations[i].field('index') == candidate.field('index')]
        # coreference_feature = len(tmp)
        # features_list.append(coreference_feature)


        # features[9]
        # appending zero feature only for noun candidates. It is set to one for pronouns
        features_list.append(0)

        # gender_feature = 0
        # if candidate.get_feature('Gender') and pronoun_info.get_feature('Gender') is not None:
        #     if features_list.append(candidate.get_feature('Gender') in pronoun_info.get_feature('Gender')):
        #         gender_feature = 1
        # features_list.append(gender_feature)


        # print len(features_list)


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
        # if not len(pronoun_info_tmp):
        #     print("not found in pronoun list", pron.field("string"))
        #     return False

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

        # NP dependancy
        # np_dep = True
        # if tmp_word is not None:
        #     np_dep = pron.field('head') != tmp_word.field('head') or \
        #                           tmp_word.field('deprel') != "nmod" or \
        #                           tmp_word.get_feature("Case") != "Gen" or \
        #                           pron.field('sentence') != candidate.field('sentence')
        #     condition_list.append(np_dep)

        for i, cond in enumerate(condition_list):
            if not cond:
                if candidate.field('index') == 1278:
                    print("stopped here!", i, pron.field('index'))
                return False
        return True

    # def output_results(self, file):
    #     for sentence in self.text.get_sent_list():
    #         for word in sentence.get_word_list():
    #             file.write(reduce((lambda x, y: x + " " + y), word.field('string')) + '\n')


    def build_pronoun_features(self, candidate, pronoun):
        pronoun_info = [i for i in self.pronoun_list if i.get_text() == pronoun.field('text')][0]
        features_list = []
        if (pronoun.antecedent_sh > 0):
            # case where antecedent might be a pronoun, which had been processed previousle (coreference)
            # then we get the features from original NP with some changes
            print("possible coreference ", pronoun.field("text"), pronoun.sh, "with", candidate.field("text"), candidate.sh)
            prev_cand = self.text.find_word(pronoun.antecedent_sh)
            if prev_cand is not None:
                features_list = self.build_features_list(prev_cand, pronoun)

            if candidate.field('index') in self.associations.keys():
                tmp = self.associations[candidate.field('index')]
                delta = pronoun.field('index') - tmp.field('index')
            else:
                delta = pronoun.field('index') - candidate.field('index')

            if delta < 0:
                delta = - delta * 3
            features_list[0] = delta

            #number of sentences between candidate and pronoun. Also might be negative
            sent_delta = 0
            if candidate.field('sentence') in self.associations.keys():
                tmp = self.associations[candidate.field('sentence')]
                sent_delta = pronoun.field('sentence') - tmp.field('sentence')
            else:
                sent_delta = pronoun.field('sentence') - candidate.field('sentence')

            if sent_delta < 0:
                sent_delta = - sent_delta * 3
            features_list[1] = sent_delta
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

    def __init__(self, filename):
        self.doc_paths = dict()
        self.answers = dict()
        self.deps = dict()

        self.parse_xml(filename)

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


sample = TrainSampleData('../corpus/groups.xml')
#

# print(list(sample.doc_paths.items())[:10])
paths = sample.doc_paths

pronoun_list = [PronounInfo(i, pronoun_feature_list[i]) for i in pronoun_text_list]

fit_paths = list(paths.items())[:200]




cls = Resolver(paths, 1, pronoun_list)
# # # #
cls.answer_dict = sample.answers

# print(sample.answers)

cls.fit(fit_paths)
# print(cls.answer_dict)
# # #
v1 = cls.predict_proba(1)
v2 = cls.predict_proba(2)
v3 = cls.predict_proba(3)
# v4 = cls.predict_proba(4)
v5 = cls.predict_proba(5)
v6 = cls.predict_proba(6)
v7 = cls.predict_proba(7)
#
# print(v2[0])
print(v1[0], v2[0], v3[0], v5[0], v6[0], v7[0])



# print(paths[-1])
# #
# fp = file_parser('/home/gand/death/Diploma/corpus/rucoref_texts/', '../corpus/Tokens.txt')
#
# fp.parse_all_files(paths)
# #
# fp.prepare_file(1)
# fp.parse_file('tmp', paths[0] + '_parsed')
#






# sample = TrainSampleData('../corpus/groups.xml')
# # #
# # # print(list(sample.doc_paths.keys())[:10])
# #
# # # #
# cls = Resolver("../corpus/rucoref_texts/" + sample.doc_paths[1] + '_parsed', 1, pronoun_list)
# # # #
# cls.answer_dict = sample.answers
# print(cls.answer_dict)
# # #
# cls.predict()


