from sklearn.cluster import KMeans
import numpy as np
import gzip
import pickle
import random

## number of classifiers
n_models = 0

## if classifier prediction doesn't match true label, returns true
def compute_error(Ci, Ti):
    return (Ti== Ci)

## updates variance of cluster k after sampling n 
def updateVar(k,n):
    global cWeights, clusters, cAccs, nks, n_models

    ## rounding and checking
    n = int(n)
    if (n == 0):
        return

    Wk = cWeights[k]
    Nk = len(clusters[k])
    Ak = cAccs[k]
    nk = nks[k]

    variance = 0
    ## accumulates variance of all classifiers
    for x in range (n_models):
        ## accuracy of x th classifier
        p = Ak[x]
        if (nk == 1):
            variance += float(nk*(p*(1.0-p)))/(nk)
        else:
            variance += float(nk*(p*(1.0-p)))/(nk-1.0)

    ## averages variances from each classifier
    variance = variance/n_models

    cVariances [k] = variance
    return variance


## calculates variance of each cluster
def calcVar():
    global cWeights, cVariances, nks

    totalVar = 0
    for k in range (K):
        varK = ((cWeights[k]**2)* (cVariances[k])/ float(nks[k])) #EQN 6, anurag's paper
        totalVar = totalVar + varK

    return totalVar

##E calculates overall accuracies of each classifier
def calcAcc():
    global cWeights, cAccs, n_models

    ## accuracies for each classifier
    totalAccs = [0 for _ in range (n_models)]

    ## for each cluster, add weighted accuracy
    for k in range (K):
        
        ## for each classifier
        for j in range (n_models):
            totalAccs[j] += cWeights[k]*cAccs[k][j]

    return totalAccs

def averageVar():
    global cVariances

    totalVar = 0
    for k in range (K):
        totalVar += cVariances[k]
    return totalVar/float(K)

## samples n instances from cluster k
def updateAccs (k,n):
    global clusters, cAccs, nks, n_models

    ## rounding and checking
    n = int(n)
    if (n == 0):
        return

    ## randomly selects n samples from cluster
    samples = random.sample(clusters[k], n)

    ## keeps track of accuracies of all classifiers
    accuracies = [0 for _ in range (n_models)]

    countC = 0
    ## sums accurate predictions
    for i in range (n):
        (V, Ci, Ti) = samples[i]

        ## for each classifier
        for j in range (n_models):
            if (Ti[j] == Ci[j]):
                accuracies[j] += 1

    ## calculates accuracy
    for j in range (n_models):
        accuracies[j] = (float(accuracies[j]) + (nks[k] * cAccs[k][j]))/(nks[k] + float(n))
    cAccs[k] = accuracies

    ## updates number of samples from cluster k
    nk = n + nks[k]
    nks[k] = nk


## computes accuracy of each classifier for each cluster
def segAcc():
    global clusters, segAccs, n_models
    for k in range (K):
        sumK = 0

        accSums = [0 for _ in range (n_models)]
        meanSums = [0 for _ in range (n_models)]

        ## calculates accuracy of each cluster
        for j in range (len(clusters[k])):
            (V, Ci, Ti) = clusters[k][j]

            ## gets accuracy of each classifier
            for model in range (n_models): 
                if (Ci[model] == Ti[model]):
                    accSums[model] += 1

        ## calculates accuracy of the cluster for each classifier

        for j in range (n_models):
            meanSums[j] = float(accSums[j])/len(clusters[k])
        segAccs[k] = meanSums


## runs iterative algorithm
def runAlg(nTotal=0, nInit=0, nStep=0):
    global K, cWeights, cVariances, nks, n_models

    ## allocates equally initially and updates variance of each cluster
    nk0 = nInit/K
    for k in range (K):
        updateAccs(k,nk0)
        updateVar(k,nk0)

    nRem = nTotal - nInit
    i = 0

    ## iteratively alocates rest of the samples
    while nRem > 0:

        nCur = min(nRem, nStep)
        nRem = nRem - nCur

        sumWkSk = 0
        ## sum of weight*std
        for k in range (K):
            sumWkSk = sumWkSk + cWeights[k] * (cVariances[k]**0.5)

        ## estimated optimal allocation
        for k in range (K):
            if sumWkSk == 0:
                nk = nCur/float(K)
            else:
                nk = int(nCur * cWeights[k]*(cVariances[k]**0.5) / sumWkSk)

            ## reestimates variance and accuracy
            updateAccs(k, nk)
            updateVar(k, nk)

        i+=1


## runs allocation algorithm on given clustering for multiple classifiers
def main(k, ntots, nints, nsteps, data, clustering, X, num_models):
    global K, vectors, true_labels, class_labels, n_models
    global nks, clusters, cWeights, cVariances, cAccs, segAccs
    n_models = num_models

    vectors = data["vectors"]
    true_labels = data["true_labels"]
    class_labels = data["class_labels"]
    NTotal = vectors.shape[0]
    K = k

    nks = {} #samples taken from kth stratum so far
    cVariances = {} #estimated variances of clusters
    clusters = clustering #the k clusters
    cWeights = {} #cluster weights
    cAccs = {} #cluster accuracy estimates
    segAccs = {}

    segAcc()
    print ("accuracies", segAccs)

    ## weight of each cluster
    for i in range (K):
        cWeights[i] = float(len(clusters[i]))/NTotal

    print ("weights", cWeights)
    results = []

    ## repeat experiment with different parameters
    for n_loop in ntots:
        for nint_loop in nints:
            for nstep_loop in nsteps:
                nTotal = n_loop
                nInit = nint_loop
                nStep = nstep_loop

                allAccs = []
                allVars = []

                ## keeps track of number of samples needed
                samples = []
                
                for x in range(X):

                    for i in range (K):
                        nks [i] = 0
                        cVariances[i] = 0 #0.25
                        cAccs[i] = [0 for _ in range (num_models)]

                    runAlg(nTotal=nTotal, nInit=nInit,nStep=nStep)

                    allAccs.append(calcAcc())
                    allVars.append(calcVar())
                    samples.append(sum(nks.values()))


                samplesUsed = np.mean(np.array(samples))
                print " finished repeating the ecperiment ", X, " times!", K, nTotal, samplesUsed, nInit, nStep

                aveAccs = np.mean(np.array(allAccs), axis = 0)
                #MEA = reduce(lambda x , y: x + y, map(lambda x: abs(x - 0.72473), allAccs))/X
                aveVar = np.mean(np.array(allVars), axis = 0)
                actVar = np.var(np.array(allAccs), axis = 0)

                print "average accuracy  : ", aveAccs
                print "actual variance   : ",  actVar
                #print "MAE               : ",  MEA
                print "average variance  : ", aveVar, "\n"

                #results.append(((aveAccs, aveVar, actVar, MEA), (nTotal, nInit, nStep, K)))
    return results
