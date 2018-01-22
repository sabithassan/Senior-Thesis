from sklearn.cluster import KMeans
import numpy as np
import gzip
import pickle
import random

def createClusters():
    global vectors, class_labels, true_labels
    global clusters, nks, cWeights, cVariances, cAccs

    print "Running KMeans"
    X = vectors
    NTotal = len(X)

    kmeans = KMeans(n_clusters = K, max_iter = 10).fit(X)
    mapped = kmeans.labels_

    print "Mapping"
    for i in range (len(mapped)):
        cluster = mapped[i]
        if cluster not in clusters:
            clusters[cluster] = [(X[i],class_labels[i],true_labels[i])]
        else:
            clusters[cluster].append((X[i],class_labels[i],true_labels[i]))

def updateVar(k,n):
    global cWeights, clusters, cAccs, nks, segAccs

    n = int(n)
    if (n == 0):
        return

    Wk = cWeights[k]
    Nk = len(clusters[k])
    Ak = cAccs[k]
    nk = nks[k]
    p = (Ak + float(segAccs[k]))/2.0
    if nk == 1:
        Vk = float(nk*(Ak*(1.0-Ak)))/(nk)
    else:
        Vk = float(nk*(Ak*(1.0-Ak)))/(nk-1.0)
    Vk_1 = Vk

    cVariances [k] = Vk_1
    return Vk_1

def calcVar():
    global cWeights, cVariances, nks

    totalVar = 0
    for k in range (K):
        if nks[k] == 0:
            varK = 0
        else:
            varK = ((cWeights[k]**2)* (cVariances[k])/ float(nks[k])) #EQN 6, anurag's paper
        totalVar = totalVar + varK

    return totalVar

#segregates accuracies within each stratum
def segAcc():
    global clusters, segAccs

    for k in range (K):
        sumK = 0
        for j in range (len(clusters[k])):
            (V, Ci, Ti) = clusters[k][j]
            if (Ci == Ti):
                sumK = sumK + 1
        mean = float(sumK)/len(clusters[k])
        segAccs[k] = mean



def calcAcc():
    global cWeights, cAccs

    totalAcc = 0
    for k in range (K):
        totalAcc = totalAcc + cWeights[k]*cAccs[k]

    return totalAcc * 100

def averageVar():
    global cVariances

    totalVar = 0
    for k in range (K):
        totalVar += cVariances[k]
    return totalVar/float(K)

def updateAccs (k,n):
    global clusters, cAccs, nks

    n = int(n)

    if (n == 0):
        return

    samples = random.sample(clusters[k], n)

    countC = 0
    for i in range (n):
        (V, Ci, Ti) = samples[i]
        if (Ti == Ci):
            countC = countC + 1

    if (nks[k] + float(n)):
        accK = (float(countC) + (nks[k] * cAccs[k]))/(nks[k] + float(n))

    else:
        accK = 0

    cAccs[k] = accK

    nk = n + nks[k]
    nks[k] = nk

def runAlg(nTotal=0, nInit=0, nStep=0):
    global K, cWeights, cVariances

    nk0 = nInit/K
    for k in range (K):
        updateAccs(k,nk0)
        updateVar(k,nk0)

    #print "Initial ACCS", cAccs
    #print "Initial Vars", cVariances

    nRem = nTotal - nInit
    i = 0
    while nRem > 0:

        nCur = min(nRem, nStep)
        nRem = nRem - nCur

        sumWkSk = 0
        for k in range (K):
            sumWkSk = sumWkSk + cWeights[k] * (cVariances[k]**0.5)

        for k in range (K):
            if sumWkSk == 0:
                nk = int(nCur/float(K))
            else:
                nk = int(nCur * cWeights[k]*(cVariances[k]**0.5) / sumWkSk)
            updateAccs(k, nk)
            updateVar(k, nk)

        i+=1

def main(k, ntots, nints, nsteps, data, clustering, X):
    global K, vectors, true_labels, class_labels
    global nks, clusters, cWeights, cVariances, cAccs, segAccs

    vectors = data["vectors"]
    true_labels = data["true_labels"]
    class_labels = data["class_labels"]
    # NTotal = len(vectors)
    NTotal = vectors.shape[0]

    K = k

    nks = {} #samples taken from kth stratum so far
    cVariances = {} #estimated variances of clusters
    clusters = {} #the k clusters
    cWeights = {} #cluster weights
    cAccs = {} #cluster accuracy estimates
    segAccs = {}

    # createClusters()

    Sk = segAcc()
    print segAccs

    for i in range (K):
        cWeights[i] = float(len(clusters[i]))/NTotal

    results = []

    for n_loop in ntots:
        for nint_loop in nints:
            for nstep_loop in nsteps:
                nTotal = n_loop
                nInit = nint_loop
                nStep = nstep_loop

                allAccs = []
                allVars = []
                samples = []

                for x in range(X):

                    for i in range (K):
                        nks [i] = 0
                        cVariances[i] = 0
                        cAccs[i] = 0

                    runAlg(nTotal=nTotal, nInit=nInit,nStep=nStep)

                    allAccs.append(calcAcc()/100.0)
                    allVars.append(calcVar())
                samples.append(sum(nks.values()))

                samplesUsed = np.mean(np.array(samples))
                print " finished repeating the ecperiment ", X, " times!", K, nTotal, samplesUsed, nInit, nStep

                aveAccs = np.mean(np.array(allAccs))
                aveVar = np.mean(np.array(allVars))
                actVar = np.var(np.array(allAccs))

                print "average accuracy  : ", aveAccs * 100
                print "actual variance   : ",  actVar
                print "average variance  : ", aveVar, "\n"

                results.append(((aveAccs, aveVar, actVar, actVar), (nTotal, nInit, nStep, K)))
    return results
