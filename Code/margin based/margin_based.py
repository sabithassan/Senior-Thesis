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
otherVecs = []
otherLabels = []

## for verification data
vSpeakerVecs = []
vSpeakerLabels = []
vOtherVecs = []
vOtherLabels = []

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
def updateSamples (classifier, vectors):

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


## trains active learner
def trainActive(kern, vectors, labels, initial, others):

    print "training active learner"
    ## chooses inital number of samples from available data
    training_samples = random.sample(range(0, len(vectors)), initial)
    training_vecs = vectors[training_samples]
    training_labels = labels[training_samples]
    active_classifier = createSVM2(kern, training_vecs, training_labels)

    print "len vec", len(training_vecs), "len labels", len(training_labels)
    ## repeatedly adds sample
    for x in range (others):
        ## gets the sample and adds to data
        sample = updateSamples(active_classifier, vectors)
        print "sample to be updated", sample
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
def readData (speakers, vecs, labels, train, target):
    global speakerFiles, speakerVecs, speakerLabels, otherVecs, otherLabels
    global vSpeakerVecs, vSpeakerLabels, vOtherVecs, vOtherLabels
    print "reading data"

    speakerFiles = {}
    # reads the target speaker.
    for x in speakers:
        getFileNames(x, train)
    filenames = []
    for x in speakers:
        filenames = filenames + speakerFiles[x]

    count = 0
    for fi in filenames:
        count = count + 1
        try:
            f = open(fi, "r")
            feat = pickle.load(f)
            vecs.append(feat)
            if (target):
                labels.append(1)
            else:
                labels.append(0)
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

## performs verification task for target speaker    
def main (target):
    global speakerFiles, speakerVecs, speakerLabels, otherVecs, otherLabels
    global vSpeakerVecs, vSpeakerLabels, vOtherVecs, vOtherLabels, mean, std
    
    #target = 101
    otherSpeakers = []
    targetSpeaker = [str(target)]
    for x in range (101, 111):
        if (x != target):
            otherSpeakers.append(str(x))

    ## TRAINING DATA
            
    ## reads supervectors of target speaker
    readData(targetSpeaker, speakerVecs, speakerLabels, True, True)    
    ## reads supervectors of all other speakers
    readData(otherSpeakers, otherVecs, otherLabels, True, False)
    # all vectors and all labels
    tAllVecs = speakerVecs + otherVecs
    tAllLabels = speakerLabels + otherLabels

    ## computes mean and standard deviation
    tAllVecs = np.array(tAllVecs)
    mean = np.mean(tAllVecs)
    std = np.std(tAllVecs)

    ## normalizes data
    tAllVecs = normalize(tAllVecs, mean, std)
    tAllLabels = np.array(tAllLabels)

    # converts into np arrays
    speakerVecs = np.array(speakerVecs)
    speakerLabels = np.array(speakerLabels)
    
    otherVecs = np.array(otherVecs)
    otherLabels = np.array(otherLabels)

    speakerVecs = normalize(speakerVecs, mean, std)
    otherVecs = normalize(otherVecs, mean, std)


    ## VERIFICATION DATA
    
    ## reads supervectors of target speaker
    readData(targetSpeaker, vSpeakerVecs, vSpeakerLabels, False, True)
    vSpeakerVecs = np.array(vSpeakerVecs)
    vSpeakerLabels = np.array(vSpeakerLabels)
    

    ## reads supervectors of all other speakers
    readData(otherSpeakers, vOtherVecs, vOtherLabels, False, False)
    vOtherVecs = np.array(vOtherVecs)
    vOtherLabels = np.array(vOtherLabels)

    ## normalizes data
    vSpeakerVecs = normalize(vSpeakerVecs, mean, std)
    vOtherVecs = normalize(vOtherVecs, mean, std)
    
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
        
        ## trains active learner and passive learner
        print "data length", len(tAllVecs)
        passive_classifier = trainPassive("linear", tAllVecs, tAllLabels, 100)
        active_classifier = trainActive("linear", tAllVecs, tAllLabels, 50, 50)
        
        ## computes erros for passive learner
        positives_passive = predict(vSpeakerVecs, passive_classifier, x)
        negatives_passive = predict(vOtherVecs, passive_classifier, x)

        frr = computeFRR(positives_passive)
        far = computeFAR(negatives_passive)

        print "false rejections passive", frr
        print "false acceptance passive", far

        positives_active = predict(vSpeakerVecs, active_classifier, x)
        negatives_active = predict(vOtherVecs, active_classifier, x)

        frr = computeFRR(positives_active)
        far = computeFAR(negatives_active)

        print "false rejections active", frr
        print "false acceptance active", far
    
## runs verification for speaker
main(101)
