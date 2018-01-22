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

    accK = (float(countC) + (nks[k] * cAccs[k]))/(nks[k] + float(n))
    cAccs[k] = accK

    nk = n + nks[k]
    nks[k] = nk

def updateVar(k,n):
    global cWeights, clusters, cAccs, nks

    n = int(n)
    if (n == 0):
        return

    Wk = cWeights[k]
    Nk = len(clusters[k])
    Ak = cAccs[k]
    nk = nks[k]

    Vk = float(nk*(Ak*(1.0-Ak)))/(nk-1.0)
    Vk_1 = ((float(nk) * Vk) + Vk)/(4.0*Vk + nk)

    cVariances [k] = Vk_1
    return Vk_1

def calcVar():
    global cWeights, cVariances, nks

    totalVar = 0
    for k in range (K):
        varK = ((cWeights[k]**2)* (cVariances[k])/ float(nks[k])) #EQN 6, anurag's paper
        totalVar = totalVar + varK

    return totalVar

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

def runAlg(nTotal=0, nInit=0, nStep=0):
    global K, cWeights, cVariances

    nk0 = nInit/K
    for k in range (K):
        updateAccs(k,nk0)
        updateVar(k,nk0)

    nRem = nTotal - nInit
    i = 0
    while nRem > 0:

        nCur = min(nRem, nStep)
        nRem = nRem - nCur
        avgVar = averageVar()

        specialStrata = []

        sumWkSk = 0
        for k in range (K):
            sumWkSk = sumWkSk + cWeights[k] * (cVariances[k]**0.5)

        for k in range (K):
            if sumWkSk == 0:
                nk = nCur/float(K)
            else:
                nk = int(nCur * cWeights[k]*(cVariances[k]**0.5) / sumWkSk)
            updateAccs(k, nk)
            updateVar(k, nk)

        i+=1

def main(k, ntots, nints, nsteps, data, X):
    global K, vectors, true_labels, class_labels
    global nks, clusters, cWeights, cVariances, cAccs

    vectors = data["vectors"]
    true_labels = data["true_labels"]
    class_labels = data["class_labels"]
    NTotal = len(vectors)
    K = k

    nks = {} #samples taken from kth stratum so far
    cVariances = {} #estimated variances of clusters
    clusters = {} #the k clusters
    cWeights = {} #cluster weights
    cAccs = {} #cluster accuracy estimates

    createClusters()

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

                for x in range(X):

                    for i in range (K):
                        nks [i] = 0
                        cVariances[i] = 0.25
                        cAccs[i] = 0

                    runAlg(nTotal=nTotal, nInit=nInit,nStep=nStep)

                    allAccs.append(calcAcc()/100.0)
                    allVars.append(calcVar())

                print " finished repeating the ecperiment ", X, " times!", K, nTotal, nInit, nStep

                aveAccs = np.mean(np.array(allAccs))
                aveVar = np.mean(np.array(allVars))
                actVar = np.var(np.array(allAccs))

                print "average accuracy  : ", aveAccs * 100
                print "actual variance   : ",  actVar
                print "average variance  : ", aveVar, "\n"

                results.append(((aveAccs, aveVar, actVar), (nTotal, nInit, nStep, K)))
    return results
