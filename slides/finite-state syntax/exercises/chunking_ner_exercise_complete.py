import nltk

from nltk.corpus import conll2000


##########################################################################
# Exercise 1: Chunking
##########################################################################

print(conll2000.chunked_sents('train.txt')[99])

""" Let's do some data exploration.
1. First, how many sentences are there?
2. How many NP chunks?
3. How many VP chunks?
4. How many PP chunks?
5. What is the average length of each?
"""
from nltk import FreqDist
from collections import defaultdict
import numpy as np

sents = conll2000.chunked_sents('train.txt')
labels = FreqDist()
lengths = defaultdict(lambda: list())

for sent in sents:
    for subtree in sent.subtrees():
        label = subtree.label()
        labels.update([label])
        lengths[label].append(len(subtree))


print()
print("######################### Exercise 1 ###############################")

print("# sents: {0}".format(len(sents)))

for label in ["NP", "VP", "PP"]:
    print(label)
    print("-"*40)
    print("# chunks: {0}".format(labels[label]))
    print("avg len: {0}".format(np.mean(lengths[label])))


##########################################################################
# Exercise 2: Unigram chunker
##########################################################################


"""
Now, let's concentrate only on NP chunking
1. Create a unigram chunker using the UnigramChunker class below.
Train on the train sentences and evaluate on the test sentences using
the evaluate method, i.e., my_model.evaluate(test_sents).


2. What is the F1 score?


"""


class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)


print()
print("######################### Exercise 2 ###############################")


train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])

unigram_chunker = UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents))

##########################################################################
# Exercise 3: Bigram/Trigram chunker
##########################################################################

"""
Now, modify the code to create Bigram and Trigram taggers

"""
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

class TrigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.TrigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

print()
print("######################### Exercise 3 ###############################")

bigram_chunker = BigramChunker(train_sents)
trigram_chunker = TrigramChunker(train_sents)

print(bigram_chunker.evaluate(test_sents))
print(trigram_chunker.evaluate(test_sents))

##########################################################################
# Exercise 4: Maximum Entropy model with features
##########################################################################

"""
Finally, we will use a maximum entropy classifier (a discriminative classifier)
to model the chunking task. Remember that discriminative classifiers attempt to
model p(y|x) directly, which allows us more freedom in what x is.
"""

def npchunk_features(sentence, i, history):
    word, pos = sentence[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
        nextword, nextpos = "<END>", "<END>"
    else:
        nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos),
            "pos+nextpos": "%s+%s" % (pos, nextpos)}


class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, max_iter=10, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)


class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w, t), c) for (w, t, c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


print()
print("######################### Exercise 4/5 ###############################")


#chunker = ConsecutiveNPChunker(train_sents)
#print(chunker.evaluate(test_sents))


##########################################################################
# Exercise 5: Add more features to get better performance
##########################################################################

"""Add in more features to get better performance"""



##########################################################################
# Exercise 6: NER Data
##########################################################################

"""
We will use a dataset of tweets annotated for Named Entities.

@inproceedings{derczynski-etal-2016-broad,
    title = "Broad {T}witter Corpus: A Diverse Named Entity Recognition Resource",
    author = "Derczynski, Leon  and
      Bontcheva, Kalina  and
      Roberts, Ian",
    booktitle = "Proceedings of {COLING} 2016, the 26th International Conference on Computational Linguistics: Technical Papers",
    month = dec,
    year = "2016",
    address = "Osaka, Japan",
    publisher = "The COLING 2016 Organizing Committee",
    url = "https://aclanthology.org/C16-1111",
    pages = "1169--1179",
    abstract = "One of the main obstacles, hampering method development and comparative evaluation of named entity recognition in social media, is the lack of a sizeable, diverse, high quality annotated corpus, analogous to the CoNLL{'}2003 news dataset. For instance, the biggest Ritter tweet corpus is only 45,000 tokens {--} a mere 15{\%} the size of CoNLL{'}2003. Another major shortcoming is the lack of temporal, geographic, and author diversity. This paper introduces the Broad Twitter Corpus (BTC), which is not only significantly bigger, but sampled across different regions, temporal periods, and types of Twitter users. The gold-standard named entity annotations are made by a combination of NLP experts and crowd workers, which enables us to harness crowd recall while maintaining high quality. We also measure the entity drift observed in our dataset (i.e. how entity representation varies over time), and compare to newswire. The corpus is released openly, including source text and intermediate annotations.",
}
"""

# load the data (1000 annotated tweets)
import json
from sklearn.metrics import f1_score

ner_data = []
for line in open("twitter_NER.json"):
    ner_data.append(json.loads(line))


# each example in ner_data is a json dictionary that contains two values we are interested in: tokens, entities

print(ner_data[6]["tokens"])
print(ner_data[6]["entities"])

# we need the training data as a list of lists, where the inner list contains tuples of (token, label) i.e., [[(token_1, label_1 ), (token_2, label_2), ...]]

# Test data should be a list of lists with the inner list having tokens

# Test labels should be a flat list of labels ['O', 'B-PER', 'I-PER', 'O'...]

ner_train_data = []
ner_test_data = []
ner_test_labels = []
for s in ner_data[:800]:
    ner_train_data.append(list(zip(s["tokens"], s["entities"])))
for s in ner_data[800:]:
    ner_test_data.append(s["tokens"])
    ner_test_labels.extend(s["entities"])

##########################################################################
# Exercise 6: Hidden Markov Model
##########################################################################

tagger = nltk.HiddenMarkovModelTagger.train(ner_train_data)

pred = tagger.tag_sents(ner_test_data)
pred_labels = [t for sent in pred for t in sent]
pred_labels = [tag for (token, tag) in pred_labels]


# Evaluate the model using the f1_score function from sklearn.metrics
# You should use macro F1 and make sure NOT TO COUNT the 'O' label

print()
print("######################### Exercise 6 ###############################")

f1 = f1_score(ner_test_labels, pred_labels, average="macro", labels=['B-LOC', 'I-ORG', 'B-PER', 'B-ORG', 'I-PER', 'I-LOC'])
print("F1 score: {0:.3f}".format(f1))


##########################################################################
# Exercise 7: Optional further exercises
##########################################################################

"""
The previous F1 is calculated at token level. However, for NER, we often calculate F1 at entity level.

Implement your own code to evaluate entity-level NER

1. You will need to calculate:
    Precision = (number of correctly predicted entities) / (number of predicted entities)

    Recall = (number of correctly predicted entites) / (number of gold entities)

    F1 = (2 * Precision * Recall) / (Precision + Recall)


2. You might want to implement a helper function which, given labels in IOB2 format return a list of all the entities.

"""

def entity_level_scores(gold, pred, labels=["PER", "LOC", "ORG", "MISC"]):
    """
    First, we'll convert all the token-level iob tags to a list of tuples
    where each tuple is (begin_index, end_index, label).
    ["B-PER", "I-PER", "O"] -> [(0, 2, "PER"), (2, 3, "O")]
    """
    true_ent_list = []
    pred_ent_list = []
    #
    ent_start = -1
    for i, label in enumerate(gold):
        if ent_start == -1:
            if label[0] == "B":
                ent_start = i
            elif label[0] == "I":
                assert 0 == 1
        else:
            if label[0] == "B":
                true_ent_list.append((ent_start, i, gold[i - 1][2:]))
                ent_start = i
            elif label == "O":
                true_ent_list.append((ent_start, i, gold[i - 1][2:]))
                ent_start = -1
    #
    ent_start = -1
    for i, label in enumerate(pred):
        if ent_start == -1:
            if label[0] == "B":
                ent_start = i
            elif label[0] == "I":
                ent_start = i
        else:
            if label[0] == "B":
                pred_ent_list.append((ent_start, i, pred[i - 1][2:]))
                ent_start = i
            elif label == "O":
                pred_ent_list.append((ent_start, i, pred[i - 1][2:]))
                ent_start = -1

    """
    Next we will count the number of true positives, false positives and
    false negatives, which we will use to calculate precision and recall.
    """
    TP, FP, FN, _TP = 0, 0, 0, 0
    for ent in true_ent_list:
        if ent in pred_ent_list:
            TP += 1
        else:
            FN += 1
    for ent in pred_ent_list:
        if ent in true_ent_list:
            _TP += 1
        else:
            FP += 1
    assert TP == _TP

    """
    Finally, we will compute precision, recall and F1
    """
    P = TP / ((TP + FP) + 1E-10)
    R = TP / ((TP + FN) + 1E-10)
    F1 = 2 * P * R / ((P + R) + 1E-10)

    return P, R, F1

print()
print("######################### Exercise 7 ###############################")

prec, rec, f1 = entity_level_scores(ner_test_labels, pred_labels)
print("Entity-level F1 score: {0:.3f}".format(f1))
