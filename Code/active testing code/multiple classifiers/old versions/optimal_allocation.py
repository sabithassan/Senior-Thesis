import numpy as np
import gzip
import pickle
import random


def updateVar(k,n):
    global cWeights, clusters, cAccs, nks, cVariances, segAccs

    n = int(n)
    if (n == 0):
        return

    Wk = cWeights[k]
    Nk = len(clusters[k])
    Ak = cAccs[k]
    nk = nks[k]
    Sk = segAccs[k]
    p = Sk
    Vk = float(Nk*(p*(1.0-p)))/(Nk-1.0)
    Vk_1 = Vk

    cVariances [k] = Vk_1
    return Vk_1


def calcVar():
    global cWeights, cVariances, nks

    totalVar = 0
    for k in range (K):
        if nks[k] != 0:
            varK = ((cWeights[k]**2)* (cVariances[k])/ float(nks[k])) #EQN 6, anurag's paper
        else:
            varK=0
        totalVar = totalVar + varK

    return totalVar

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

    nk0 = 0
    for k in range (K):
        #updateAccs(k,1)
        updateVar(k,1)

    nCur = nTotal
    sumWkSk = 0
    for k in range (K):
            sumWkSk = sumWkSk + cWeights[k] * (cVariances[k]**0.5)


    for k in range (K):
        if sumWkSk == 0:
            print "ME IS CRAAZY"
            nk = nCur/float(K)
        else:
            nk = int(nCur * cWeights[k]*(cVariances[k]**0.5) / sumWkSk)
        if nk == 0:
            updateAccs(k, 1)
        else:
            updateAccs(k, nk)
        #updateVar(k, nk)



def main(model, k, ntots, X, nints, nsteps, test_matrix, test_tl, clustering):
    global K
    global nks, clusters, cWeights, cVariances, cAccs, segAccs

    nks = {} #samples taken from kth stratum so far
    cVariances = {} #estimated variances of clusters
    clusters = clustering #the k clusters
    cWeights = {} #cluster weights
    cAccs = {} #cluster accuracy estimates
    segAccs = {}
    K= k


    segAcc()
    print "Accuracies:\n", segAccs


    true_accuracy = np.mean(np.array(map(lambda x, y: int(model.predict(x)==y),
            test_matrix, test_tl)))

    print "true accuracy     :", true_accuracy

    for i in range (K):
        cWeights[i] = float(len(clusters[i])) / sum(map(len, clusters.values()))
    print "Weights:\n", cWeights

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

                realn = sum(nks.values())
                print " finished repeating the experiment ", X, " times!", K, nTotal, realn, nInit, nStep

                accuracy = np.mean(np.array(allAccs))
                MAE = sum(map(lambda x: abs(x - true_accuracy), allAccs)) / X
                estimated_variance = np.mean(np.array(allVars))
                variance = np.var(np.array(allAccs))

                print "Accuracy               :", accuracy
                print "Variance               :", variance
                print "Estimated Variance     :", estimated_variance
                print "MAE                    :", MAE
                print "________________________"

                results.append(((accuracy, estimated_variance, variance, MAE), (nTotal, nInit, nStep, K)))
    return results
