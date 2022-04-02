import nltk
nltk.download("brown")
nltk.download("universal_tagset")
from nltk.corpus import brown
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
#from collections import defaultdict



def plot_confusion_matrix(cm,
                          labels,
                          cmap=plt.cm.BuPu):
    """
    This function plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    x_tick_marks = np.arange(len(labels))
    y_tick_marks = np.arange(len(labels))
    plt.xticks(x_tick_marks, labels, rotation=45)
    plt.yticks(y_tick_marks, labels)
    #
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


##########################################################
# Include your code to train Ngram taggers here
##########################################################
for domain in brown.categories():
                
    # Create train/dev/test splits
    
    taggedS = brown.tagged_sents(categories=domain, tagset="universal")
    
    trainIdx = int(len(taggedS) * .7)
    devIdx = int(len(taggedS) * .8)
    
    train = taggedS[:trainIdx]
    dev = taggedS[trainIdx:devIdx]
    test = taggedS[devIdx:]
    
    

    # train the Ngram taggers
    
    tagged0 = nltk.DefaultTagger('NOUN')
    tagged1 = nltk.UnigramTagger(train, backoff = tagged0)
    tagged2 = nltk.BigramTagger(train, backoff = tagged1)
    tagged3 = nltk.TrigramTagger(train, backoff = tagged2)

    # get the dev and test accuracy and their difference
    
    testAccuracy = 100.0 * tagged3.evaluate(test)
    devAccuracy = 100.0 *  tagged3.evaluate(dev)
    
    differenceEvaluations = devAccuracy-testAccuracy
    
    # print out the domain/dev acc/test acc/ and difference
    
    
    
    
    print("Train len for",domain,"domain: ",len(train))
    print("Dev len for",domain,"domain: ",len(dev))
    print("Test len for",domain,"domain: ",len(test))


##########################################################
# Include your code to train Ngram taggers here
##########################################################

# convert test to the correct data format (a flat list of tags)

taglessTList = list()


for sentence in test:
  taglessTList.append([tag[0] for tag in sentence])

correctTdata = [tag[1] for sentence in test for tag in sentence]

Tags = []
for sentence in test:
    for word in sentence:
        wrd, tag = word
        Tags.append(tag)

# remove the tags from the original test data and use tag_sents() to get the predictions from the final model

untagged_test = []
for sentence in test:
    Newsent = []
    for word in sentence:
        wrd, tag = word
        Newsent.append(wrd)   
    untagged_test.append(Newsent)
    
prediction_test = tagged3.tag_sents(untagged_test)

# convert the predictions to the correct data format (a flat list of predicted tags)


convert_predictions = [tag[1] for pair in prediction_test for tag in pair]

labels = sorted(set(Tags))

# create the confusion matrix and plot it using the plot_confusion_matrix function

        
confusion_m = confusion_matrix(correctTdata, convert_predictions, labels = labels)
plot_confusion_matrix(confusion_m, labels = labels)

