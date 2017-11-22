import pickle
from sklearn.svm import LinearSVC
import numpy as np
a = np.array([1,2,3,4])
print ("a shape", a.shape)

## reads features of file
def getFeatures(filename):
    file = open(filename,'r')
    print (file)
    features = []
    features1 = pickle.load(file)
    for x in range (len(features1)):
        ## flattens it cause cnn features are (1,1024). changes to (1024)
        features.append( features1[x].ravel())
    print (len(features1))
    file.close()
    return features

## reads labels of files
def getLabels(filename):
    file = open(filename,'r')
    print (file)
    labels1 = pickle.load(file)
    ##print (featr)
    print (len(labels1))
    file.close()
    return labels1

feats1 = getFeatures("5100feats.pkl")
labels1 = getLabels("5100labels.pkl")

feats2 = getFeatures("10100feats.pkl")
labels2 = getLabels ("10100labels.pkl")

testFeats = getFeatures("15100feats.pkl")
testLabels = getLabels("15100labels.pkl")

testFeats = np.array(testFeats)
testLabels = np.array(testLabels)

trainingFeats = feats1 + feats2
trainingFeats = np.array(trainingFeats)

trainingLabels = labels1 + labels2
trainingLabels = np.array(trainingLabels)

print ("training shape", trainingFeats.shape)
print ("training label shape", trainingLabels.shape)
print ("testing shape", testFeats.shape)
print ("testing label shape", testLabels.shape)



from sklearn import svm
import numpy as np
import math
import pickle
import random
import matplotlib.pyplot as plt

## file names
speakerFiles = {}

## for training data
speakerVecs = []
speakerLabels = []
allVecs = []
allLabels = []

## for verification data
vSpeakerVecs = []
vSpeakerLabels = []
vAllVecs = []
vAllLabels = []

## keeping track globally
allInd = []

# computes FAR error
def computeFAR(labels):
    return 100*sum(labels)/float(len(labels))


#computes FRR error
def computeFRR (labels):
    return 100*(len(labels) - sum(labels))/float(len(labels))


#trains SVM 
def createSVM(kern, vectors, labels):

    clf = svm.SVC(kernel = kern, probability = True)
    clf.fit(vectors, labels)

    return clf


#trains SVM with probability
def createSVM2(kern, vectors, labels):
    
    clf = svm.SVC(kernel = kern, probability = True)
    clf.fit(vectors, labels)

    return clf

## returns difference between most likely and second most likely
def getMargin(classProbs):
    
    sortedProbs = np.sort(classProbs)
    flipped = sortedProbs[::-1]
    if (len(sortedProbs)<2):
        print "There should be at least two classes"

    return flipped[0]-flipped[1]


#returns sample with lowest margin difference
def updateMargin (classifier, vectors):

    global allInd
    probs = classifier.predict_proba(vectors)
    min_diff = 999999
    ind = -1

    ## checks all data for sample with lowest margin difference
    for x in range (len(probs)):
        classProbs = probs[x]
        margin = getMargin(classProbs)
        if (margin < min_diff):
            if x not in allInd:
                min_diff = margin
                ind = x
                
    if (ind == -1):
        print "INDEX IS -1, something wrong"
    allInd.append(ind)
    return ind

## gets sample based on least confidence
def updateLeastConf(classifier, vectors):
    global allInd
    probs = classifier.predict_proba(vectors)
    max_diff = -9999999
    for x in range (len(probs)):
        classProbs = probs[x]
        sortedProbs = np.sort(classProbs)
        flipped = sortedProbs[::-1]
        diff = 1- flipped[0]
        if (diff > max_diff):
            if x not in allInd:
                max_diff = diff
                ind = x
                
    if (ind == -1):
        print "INDEX IS -1, something wrong"
    allInd.append(ind)
    return ind

## gets sample based on maximum entropy
def updateEntropy(classifier, vectors):
    global allInd
    probs = classifier.predict_proba(vectors)
    max_diff = -9999999
    for x in range (len(probs)):
        classProbs = probs[x]
        sumProbs = 0
        ## calculates entropy:
        for prob in classProbs:
            if (not (prob == 0)):
                sumProbs += prob*math.log(prob)
        sumProbs = -sumProbs
        
        if (sumProbs > max_diff):
            if x not in allInd:
                max_diff = sumProbs
                ind = x
                
    if (ind == -1):
        print "INDEX IS -1, something wrong entropy"
    allInd.append(ind)
    return ind


## updates sample based on maxmin
def updateMaxMin(classifier, vectors):
    return



## trains active learner
def trainActive(kern, vectors, labels, initial, others, measure):
    global allInd
    allInd = []
    
    print "training active learner, ", measure
    ## chooses inital number of samples from available data
    training_samples = random.sample(range(0, len(vectors)), initial)
    training_vecs = vectors[training_samples]
    training_labels = labels[training_samples]
    active_classifier = createSVM2(kern, training_vecs, training_labels)

    print "len vec", len(training_vecs), "len labels", len(training_labels)
    ## repeatedly adds sample
    for x in range (others):
        ## gets the sample and adds to data
        if (measure == "margin"):
            sample = updateMargin(active_classifier, vectors)
        elif (measure == "least_conf"):
            sample = updateLeastConf(active_classifier, vectors)
        elif (measure == "entropy"):
            sample = updateEntropy(active_classifier, vectors)
        else:
              print "Measure not known"
              return active_classifier
              
        # print "sample to be updated", sample
        training_vecs = np.append(training_vecs, np.array([vectors[sample]]), axis = 0)
        training_labels = np.append(training_labels,labels[sample])
        ## updates classifier after each sample
        active_classifier = createSVM2(kern, training_vecs, training_labels)
    return active_classifier


## trains passive learner
def trainPassive(kern, vectors, labels, resource):

    print "Training passive learner"
    ## chooses samples according to resource
    training_samples = random.sample(range(0, len(vectors)), resource)
    training_vecs = vectors[training_samples]
    training_labels = labels[training_samples]

    return createSVM(kern, training_vecs, training_labels)




# adds bias to decision value of SVM and returns prediction
def predict(vectors, clf, T):

    labels = clf.decision_function(vectors)
    labels = np.subtract(labels, T)
    ## if distance > 0, class 1, else 0
    f = lambda x : 1 if x >= 0 else 0
    ff = np.vectorize(f)
    decisions = ff(labels)
    
    return decisions


def main ():
    N = 500
    NINIT = 300
    NOTHER = N-NINIT
    global trainingFeats, trainingLabels, testFeats, testLabels
    ## trains active learner and passive learner
    #print "data length", len(tAllVecs)
    passive_classifier = trainPassive("linear", trainingFeats, trainingLabels, N)
    passive_score = passive_classifier.score(testFeats, testLabels)
    
    ## trains different active classifiers based on measures of uncertainty
    active_classifier_entropy = trainActive("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "entropy")
    active_classifier_least = trainActive("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "least_conf")
    active_classifier_margin = trainActive("linear", trainingFeats, trainingLabels, NINIT, NOTHER, "margin")

    ## scores of different active learners
    active_score_entropy = active_classifier_entropy.score(testFeats, testLabels)
    active_score_least = active_classifier_least.score(testFeats, testLabels)
    active_score_margin = active_classifier_margin.score(testFeats, testLabels)

    print "passive learner score", passive_score
    print "parameters", N, NINIT, NOTHER
    print "active learner entropy score", active_score_entropy
    print "active learner least confident score", active_score_least
    print "active learner margin score", active_score_margin

main()
