from sklearn.cluster import KMeans
import numpy as np
import gzip
import pickle
import random

def updateVar(k,n):
    global cWeights, clusters, cAccs, nks

    n = int(n)
    if (n == 0):
        return

    Wk = cWeights[k]
    Nk = len(clusters[k])
    Ak = cAccs[k]
    nk = nks[k]
    Sk = segAccs[k]
    p = (Ak + float(Sk))/2.0
    Vk = float(nk*(p*(1.0-p)))/(nk-1.0)
    Vk_1 = Vk

    cVariances [k] = Vk_1
    return Vk_1

def calcVar():
    global cWeights, cVariances, nks

    totalVar = 0
    for k in range (K):
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

    accK = (float(countC) + (nks[k] * cAccs[k]))/(nks[k] + float(n))
    cAccs[k] = accK

    nk = n + nks[k]
    nks[k] = nk

def runAlg(nTotal=0, nInit=0, nStep=0):
    global K, cWeights, cVariances

    nk0 = nTotal/K
    for k in range (K):
        updateAccs(k,nk0)
        updateVar(k,nk0)

def main(k, ntots, nints, nsteps, data, clustering, X):
    global K, vectors, true_labels, class_labels
    global nks, clusters, cWeights, cVariances, cAccs, segAccs

    vectors = data["vectors"]
    true_labels = data["true_labels"]
    class_labels = data["class_labels"]
    NTotal = len(vectors)
    K = k

    nks = {} #samples taken from kth stratum so far
    cVariances = {} #estimated variances of clusters
    clusters = clustering #the k clusters
    cWeights = {} #cluster weights
    cAccs = {} #cluster accuracy estimates
    segAccs = {}

    segAcc()

    print "printing accuracies", segAccs

    for i in range (K):
        cWeights[i] = float(len(clusters[i]))/NTotal

    print "printing weights", cWeights

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
                        cVariances[i] = 0
                        cAccs[i] = 0

                    runAlg(nTotal=nTotal, nInit=nInit,nStep=nStep)

                    allAccs.append(calcAcc()/100.0)
                    allVars.append(calcVar())

                print " finished repeating the ecperiment ", X, " times!", K, nTotal, nInit, nStep

                aveAccs = np.mean(np.array(allAccs))
                MEA = reduce(lambda x , y: x + y, map(lambda x: abs(x - 0.72473), allAccs))/X
                aveVar = np.mean(np.array(allVars))
                actVar = np.var(np.array(allAccs))

                print "average accuracy  : ", aveAccs * 100
                print "actual variance   : ",  actVar
                print "MAE               : ",  MEA
                print "average variance  : ", aveVar, "\n"

                results.append(((aveAccs, aveVar, actVar, MEA), (nTotal, nInit, nStep, K)))
    return results
