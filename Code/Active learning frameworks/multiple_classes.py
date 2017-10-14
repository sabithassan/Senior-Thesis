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

## gets file names for speakers
def getFileNames(k, train):
    if (train):
        s = "../Data/lists/" + k + ".traininglist"
    else:
        s = "../Data/lists/" + k + ".testlist"
    filelists = []
    f = open(s, 'r')
    filename =  f.readline().strip()
    filename = "../Data/" + filename[:len(filename)-4] + ".pkl"
    while (filename):
        filelists.append(filename)
        filename = f.readline().strip()
        if (filename):
            filename = "../Data/" + filename[:len(filename)-4] + ".pkl"
    speakerFiles [k] = filelists


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


# reads training data.
def readData (speakers, vecs, labels, train):
    global speakerFiles, speakerVecs, speakerLabels, allVecs, allLabels
    global vSpeakerVecs, vSpeakerLabels, vAllVecs, vAllLabels
    print "reading data"

    speakerFiles = {}
    # reads the speakers.
    for x in speakers:
        getFileNames(x, train)
    filenames = []
    for x in speakers:
        filenames = filenames + speakerFiles[x]

    count = 0
    ## reads all files
    for fi in filenames:
        count = count + 1
        try:
            f = open(fi, "r")
            feat = pickle.load(f)
            vecs.append(feat)
            found = False
            count = 0
            ## appends labels
            for x in speakers:
                if x in fi[:20]:
                    labels.append(x)
                    count += 1
                    found = True
                    
            ## label should be found
            if (count != 1):
                print fi
            if (not found):
                print "label not found"
            
            f.close()
        except:
            print "Shouldnt be happening"
            continue

    print "done reading data"


## plots det curve
def DETCurve(fps,fns, fps2, fns2, speaker):
    f1 = plt.figure("linear" + speaker)
    plt.plot(fps,fns, label = "linear")
    f2 = plt.figure("RBF" + speaker)
    plt.plot(fps2,fns2, label = "rbf")
    plt.show()

## normalizes data
def normalize(X, mean, std):
    X = np.divide(np.subtract(X, mean), std)
    return X

## performs speaker identification task    
def main ():
    global speakerFiles, speakerVecs, speakerLabels, allVecs, allLabels
    global vSpeakerVecs, vSpeakerLabels, vAllVecs, vAllLabels, mean, std
    

    allSpeakers = []
    for x in range (101, 111):
        ##if (x != target):
        allSpeakers.append(str(x))

    print "all speakers", allSpeakers
    ## TRAINING DATA
            
    ## reads supervectors of all speakers
        
    readData(allSpeakers, allVecs, allLabels, True)
    # all vectors and all labels
    tAllVecs = allVecs
    tAllLabels = allLabels

    ## computes mean and standard deviation
    tAllVecs = np.array(tAllVecs)
    mean = np.mean(tAllVecs)
    std = np.std(tAllVecs)

    ## normalizes data
    tAllVecs = normalize(tAllVecs, mean, std)
    tAllLabels = np.array(tAllLabels)
    
    allVecs = np.array(allVecs)
    allLabels = np.array(allLabels)

    #speakerVecs = normalize(speakerVecs, mean, std)
    allVecs = normalize(allVecs, mean, std)

    print "train", len(allVecs), len(allLabels)
    ## VERIFICATION DATA
    ## reads supervectors of all speakers
    print "ver", len(vAllVecs), len(vAllLabels)
    
    readData(allSpeakers, vAllVecs, vAllLabels, False)
    vAllVecs = np.array(vAllVecs)
    vAllLabels = np.array(vAllLabels)

    print "ver", len(vAllVecs), len(vAllLabels)

    ## normalizes data
    #vSpeakerVecs = normalize(vSpeakerVecs, mean, std)
    vAllVecs = normalize(vAllVecs, mean, std)
    
    print "donee reading all data"
    ##return
    ## The valies of Ts to bs tried:
    vals = np.arange(-2, 2.05, 0.005)
    T1 = [0]

    FARS = []
    FRRS = []
    diff = 999
    EER = -9
    bestFar = 99
    bestFrr = 99
    for x in T1:

        ## sample distribution scheme
        N = 50
        NINIT = 20
        NOTHER = N-NINIT
        
        ## trains active learner and passive learner
        print "data length", len(tAllVecs)
        passive_classifier = trainPassive("linear", tAllVecs, tAllLabels, N)
        passive_score = passive_classifier.score(vAllVecs, vAllLabels)
        
        ## trains different active classifiers based on measures of uncertainty
        active_classifier_entropy = trainActive("linear", tAllVecs, tAllLabels, NINIT, NOTHER, "entropy")
        active_classifier_least = trainActive("linear", tAllVecs, tAllLabels, NINIT, NOTHER, "least_conf")
        active_classifier_margin = trainActive("linear", tAllVecs, tAllLabels, NINIT, NOTHER, "margin")

        ## scores of different active learners
        active_score_entropy = active_classifier_entropy.score(vAllVecs, vAllLabels)
        active_score_least = active_classifier_least.score(vAllVecs, vAllLabels)
        active_score_margin = active_classifier_margin.score(vAllVecs, vAllLabels)

        print "passive learner score", passive_score
        print "parameters", N, NINIT, NOTHER
        print "active learner entropy score", active_score_entropy
        print "active learner least confident score", active_score_least
        print "active learner margin score", active_score_margin
        

    
main()
