# -*- coding: utf-8 -*-


#TODO: implement correct analysis of imput sequence shift. Special attention to punctuation marks

import subprocess
from gensim.models import KeyedVectors
from nltk.stem.snowball import SnowballStemmer
from pymystem3 import Mystem
from nltk.tokenize import *
# from sklearn import linear_model
from math import log

f = open("./parsed/output1.txt")
file = f.read()
f = file.split('\n')

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
        self.predicted = False      #for pronouns only

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

    def __init__(self, file_name):

        self.word_id = 0
        self.sent_id = 0

        print(file_name)

        self.word_sh = 0

        self.file = open(file_name, 'rt')

        self.tokens_file = open("/home/gand/death/Diploma/corpus/Tokens.txt", 'rt')

        # for excessive reading of the first word of the next sentence
        self.__tmp_first_word = None

        # ignore the first line
        # print(self.tokens_file.readline())
        sentences = []
        while len(sentences) < 10:
            tmp = self.read_sentence()
            if tmp is not None:
                sentences.append(tmp)
            else:
                break

        self.text = Text(sentences)
        return

    def read_sentence(self):

        tmp_sent = []
        if self.__tmp_first_word != None:
            tmp_sent.append(self.__tmp_first_word)

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

            tokens_line = self.tokens_file.readline()
            tok_spl = tokens_line.split()


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
                print("mismatch! ", self.word_sh, tmp_word['text'], tok_spl[3].lower(), "doc", tok_spl[0])

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


    def forward(self):
        sentence = self.read_sentence()
        self.text.forward(sentence)

    def get_text(self):
        return self.text




class Resolver:

    def __init__(self, input_file_name, pronoun_list):

        self.text_builder = TextBuilder(input_file_name)

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

    def build_prediction_list(self):
        self.pred_list = []
        for sentence in self.text.get_sent_list():
            for word in sentence.get_word_list():
                if word.field('text').lower() in pronoun_text_list and not word.predicted:
                    self.pred_list.append(word)

    def predict(self):
        i = 0.0
        self.not_founders = 0
        while len(self.text.get_sent_list()) > 0:
            self.build_prediction_list()
            for pronoun in self.pred_list:
                tmp = self.predict_word(pronoun.sh)
                if tmp is None:
                    continue
                print(pronoun.field('text'), pronoun.sh, "refers to", tmp.field('text'), tmp.sh)
                if len(self.answer_dict) > 0 and pronoun.sh in self.answer_dict[1]:
                    if self.answer_dict[1][pronoun.sh] == tmp.sh:
                        i += 1.0
                    else:
                        print("wrong! ", tmp.sh, "instead of", self.answer_dict[1][pronoun.sh])
            self.text_builder.forward()
            self.text = self.text_builder.get_text()

            # for s in self.text.get_sent_list():
            #     print(s.index)
            # print ("after_up_end\n")

        return (i, len(self.answer_list))

    def build_candidates_list(self, pronoun, length):
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
            self.features[candidate] = self.build_features_list(candidate, pronoun)

        antecedent = self.get_right_word(pronoun)
        self.associations[pronoun] = antecedent

        pronoun.field('string')[9] = "refto_" + str(antecedent.sh + 1)

        pronoun.predicted = True

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
            print("beeen", self.been_candidate)

    def build_features_list(self, candidate, pronoun):

        pronoun_info = [i for i in self.pronoun_list if i.get_text() == pronoun.field('text')][0]
        features_list = []

        # distance between candidate and pronoun. Might be negative, if candidate is further then the pronoun
        delta = 0
        if candidate.field('index') in self.associations.keys():
            tmp = self.associations[candidate.field('index')]
            delta = pronoun.field('index') - tmp.field('index')
        else:
            delta = pronoun.field('index') - candidate.field('index')

        if delta < 0:
            delta = - delta * 3
        features_list.append(delta)

        #number of sentences between candidate and pronoun. Also might be negative
        sent_delta = 0
        if candidate.field('sentence') in self.associations.keys():
            tmp = self.associations[candidate.field('sentence')]
            sent_delta = pronoun.field('sentence') - tmp.field('sentence')
        else:
            sent_delta = pronoun.field('sentence') - candidate.field('sentence')

        if sent_delta < 0:
            sent_delta = - sent_delta * 3
        features_list.append(sent_delta)

        #feature connected with candidates position in a sentence
        pos_feature = 0
        if candidate.field('deprel') == 'nsubj' or candidate.field('postag') == 'nsubjpass':
            pos_feature = 2
        elif candidate.field('deprel') == 'dobj':
            pos_feature = 1

        features_list.append(pos_feature)

        # :TODO frequency feature might be implemented
        # :TODO add word2vec feature if there will be any time left

        # feature connected with the case of pronoun and it's antecedent
        # It's written like "in" here, because each pronoun may have several  variants of case
        case_feature = 0
        if candidate.get_feature('Case') is not None:
            if candidate.get_feature('Case') in pronoun_info.get_feature('Case'):
                case_feature = 1
        features_list.append(case_feature)

        #feature connected with animacy of candidate
        animacy_feature = 0
        if pronoun.get_feature('Animacy') is not None and candidate.get_feature('Animacy') is not None:
            if candidate.get_feature('Animacy') in pronoun_info.get_feature('Animacy'):
                animacy_feature = 1
        features_list.append(animacy_feature)


        #simple parallelism feature
        synt_parallel_feature = 0
        if candidate.field("deprel") == pronoun.field("deprel"):
            synt_parallel_feature = 1
            # print("look for pronoun", pronoun.field('text'), pronoun.field('index') , "in sentence", pronoun.field('sentence') )
            head_pron = self.text.get_sentence(pronoun.field('sentence')).find_in_sentence(pronoun.field('head'))
            head_candidate = self.text.get_sentence(candidate.field('sentence')).find_in_sentence(candidate.field('head'))
            if head_candidate is not None and head_pron is not None and head_candidate.field('deprel') == head_pron.field('deprel'):
                synt_parallel_feature = 3
                if head_pron.field('deprel') == 'ROOT':
                    synt_parallel_feature = 6
        features_list.append(synt_parallel_feature)

        #Frequency feature
        frequency_feature = self.get_word_frequency(self.mystem.lemmatize(candidate.field('text'))[0])
        # frequency_feature = self.text.get_word_frequency(self.stemmer.stem(candidate.field('text').decode('utf8')).encode('utf8'))
        features_list.append(frequency_feature)

        #Using Word2Vec
        similarity_feature = 0
        left_neighbour = self.text.find_word_by_id(pronoun.field('index') - 1)

        if left_neighbour.field('punct text') != '_':
            left_neighbour = self.text.find_word_by_id(pronoun.field('index') - 2)

        right_neighbour = self.text.find_word_by_id(pronoun.field('index') + 1)
        if right_neighbour.field('punct text') != '_':
            right_neighbour = self.text.find_word_by_id(pronoun.field('index') + 2)

        try:
            left_neighbour_lemma = self.mystem.lemmatize(left_neighbour.field('text'))[0]
            right_neighbour_lemma = self.mystem.lemmatize(right_neighbour.field('text'))[0]

            candidate_lemma = self.mystem.lemmatize(candidate.field('text'))[0]

            if left_neighbour_lemma is not None:
                similarity_feature += self.model.similarity(candidate_lemma.decode('utf8'),
                                                            left_neighbour_lemma.decode('utf8'))
            if right_neighbour_lemma is not None:
                similarity_feature += self.model.similarity(candidate_lemma.decode('utf8'),
                                                            right_neighbour_lemma.decode('utf8'))
        except:
            pass

        features_list.append(similarity_feature)

        tmp = [i for i in self.associations.keys() if self.associations[i].field('index') == candidate.field('index')]
        coreference_feature = len(tmp)
        features_list.append(coreference_feature)

        # gender_feature = 0
        # if candidate.get_feature('Gender') and pronoun_info.get_feature('Gender') is not None:
        #     if features_list.append(candidate.get_feature('Gender') in pronoun_info.get_feature('Gender')):
        #         gender_feature = 1
        # features_list.append(gender_feature)


        # print len(features_list)

        return features_list

    def is_word_acceptable(self, pron, candidate):

        condition_list = []

        # leave nouns only

        if candidate.field('postag') != 'NOUN':
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

    def output_results(self, file):
        for sentence in self.text.get_sent_list():
            for word in sentence.get_word_list():
                file.write(reduce((lambda x, y: x + " " + y), word.field('string')) + '\n')


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

            anaph_groups = doc.findall(".//attributes/*[@val='anaph']../..")
            self.deps[id] = dict()

            for group in anaph_groups[:100]:
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
                        head_item = [i for i in tmp_ref.findall(".//*[@gram]") if i.get('gram') != '-'][0]
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

        print("len = ", len(tokens))

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


    def parse_all_files(self, files_list):
        for i, file in enumerate(files_list):
            self.prepare_file(file)
            self.parse_file('tmp', file + "_parsed")
            print("parsed file", i)


sample = TrainSampleData('../corpus/groups.xml')
#
paths = list(sample.doc_paths.values())

# print(paths[-1])
# #
# fp = file_parser('/home/gand/death/Diploma/corpus/rucoref_texts/', '../corpus/Tokens.txt')
# #
# fp.prepare_file(1)
# fp.parse_file('tmp', paths[0] + '_parsed')
#
# fp.parse_all_files(paths)


sample = TrainSampleData('../corpus/groups.xml')
# #
# # print(list(sample.doc_paths.keys())[:10])
#
pronoun_list = [PronounInfo(i, pronoun_feature_list[i]) for i in pronoun_text_list]
# # #
cls = Resolver("../corpus/rucoref_texts/" + sample.doc_paths[1] + '_parsed', pronoun_list)
# # #
cls.answer_dict = sample.answers
print(cls.answer_dict)
# #
cls.predict()


