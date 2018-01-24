import gzip, pickle, random, time, sys, clustering, os
import scipy, sklearn

import numpy as np
import random_sampling as SRS
import optimalAlloc as opt

import equalAlloc as eq
import OPT_A4_VAR as KS4
import OPT_A4_STD as KS41
import OPT_A3 as KS3
import OPT_A2 as KS2
import OPT_A5 as KS5

from sklearn.datasets import load_svmlight_file
from sklearn import svm
from datetime import date
from sklearn.svm import LinearSVC
from sets import Set
from copy import copy
from collections import Counter
from sklearn.externals import joblib
from sklearn.svm import LinearSVC

num = sys.argv[3]

train_file = sys.argv[1]
test_file = sys.argv[2]


num = 3

print "Start reading training file"
train_matrix, train_tl = load_svmlight_file(train_file)
print "Finished reading training file"
print "Start training classifier"
linear_SVC = svm.LinearSVC()
linear_SVC.fit(train_matrix, train_tl)
my_svms = linear_SVC
print "Finished training classifier"

del train_matrix
del train_tl



print "Start reading testing file"
def get_data(path):
    features, labels = load_svmlight_file(path)
    # features, labels = load_svmlight_file(path, n_features=2001, length=2000)


    print features.shape
    print labels.shape
    return (features, labels)

test_matrix, test_tl = load_svmlight_file(test_file)
# (X_testing_CNN, strong_labels) = get_data(("../datasets/epsilon_normalized.t.bz2"))
# X_testing_CNN = scipy.sparse.csr_matrix.todense (X_testing_CNN)
# test_matrix, test_tl = X_testing_CNN, strong_labels

# my_svms = joblib.load('../FAR_vs_FRR/svm.pkl')
# unique_labels = [-1., 1.]

print "Finished reading testing file"

nTotals = range(20, 61, 10)
K = [6, 7]
nInit = [0]
nStep = [150]
X = 2500



model = my_svms

score = 0# model.score(test_matrix,test_tl)
print "True Accueacy of classifier        ", score
data = {"vectors" : test_matrix, "true_labels":test_tl, "class_labels":score}



random_outfile = "data/random_"+date.today().strftime("%d_%m_%y")+"_"+str(num)+".data"
stratified_outfile = "data/stratified_"+date.today().strftime("%d_%m_%y")+"_"+str(num)+".data"


# test_scores = map(model.decision_function, test_matrix)


def run_random_sampling():
    results = SRS.main(nTotals, X, model, test_matrix, test_tl)
    pickle.dump(results, open(random_outfile, "a"))
    return results

def run_stratified_sampling():

    k2, k3, k4, k41 = [], [], [], []
    optim, equal, ops_optim = [], [], []

    for k in K:
        clusters = clustering.KMeanss(model, k, test_matrix, test_tl)

        print "OPTIMAL"
        d = opt.main(k, nTotals, nInit, nStep, data, clusters, X)
        optim.append(d)
        print "___________________________________________________"

        print "OPT_A2"
        d = KS2.main(k, nTotals, nInit, nStep, data, clusters, X)
        k2.append(d)
        print "___________________________________________________"

        # print "EQUAL"
        # d = eq.main(k, nTotals, nInit, nStep, data, clusters, X)
        # equal.append(d)
        # print "___________________________________________________"

        print "OPT_A3"
        d = KS3.main(k, nTotals, nInit, nStep, data, clusters, X)
        k3.append(d)
        print "___________________________________________________"

        print "OPT_A4 threshold"
        d = KS4.main(k, nTotals, nInit, nStep, data, clusters, X)
        k4.append(d)
        print "___________________________________________________"

        print "OPT_A4 std"
        d = KS41.main(k, nTotals, nInit, nStep, data, clusters, X)
        k41.append(d)
        print "___________________________________________________"

        print "OPT_A5"
        d = KS5.main(k, nTotals, nInit, nStep, data, clusters, X)
        ops_optim.append(d)
        print "___________________________________________________"


    return (k2, k3, k4, k41, optim, equal, ops_optim)

# This function is used for sorting lists using QuickSort algorithm. This was
#  retrieved from the internet.
def quicksort( aList):

    def swap(A, x, y):
    	A[x],A[y]=A[y],A[x]

    def partition(aList, first, last):
        pivot = first + random.randrange(last - first + 1)
        swap(aList, pivot, last)
        for i in range(first, last):
        	if GE(aList[i], aList[last]):
    			swap(aList, i, first)
    			first += 1
        swap(aList, first, last)
        return first

    def GE (a, b):
        return a[1] <= b[1]

    def _quicksort( aList, first, last,):
    	if first < last:
    		pivot = partition(aList, first, last)
    		_quicksort(aList, first, pivot - 1)
    		_quicksort(aList, pivot + 1, last)

	_quicksort( aList, 0, len( aList ) - 1)


def dict_to_list(d):

    def helper(k, d):
        a = d[k]
        quicksort(a)
        r = map(lambda (x, y) : x, a)
        return (k, r)

    return map(lambda k: helper(k, d), d.keys())

# This function is used to collect the data that is outputed from the function
#  stratifiedKMeans.
def collect(strat_d):
    # print strat_d
    cestimated_vars = {}
    result_acc = {}
    result_truevar = {}
    MAEs = {}
    seen = []
    for i in range(len(strat_d)):
        for ((cacc, cestimated_var, cvar, MAE), (n, nint, nstep, k)) in strat_d[i]:
            if (nint, nstep, k) in seen:
                cestimated_vars[(nint, nstep, k)].append((cestimated_var, n))
                result_truevar[(nint, nstep, k)].append((cvar, n))
                result_acc[(nint, nstep, k)].append((cacc, n))
                MAEs[(nint, nstep, k)].append((MAE, n))
            else:
                cestimated_vars[(nint, nstep, k)] = [(cestimated_var, n)]
                result_truevar[(nint, nstep, k)] = [(cvar, n)]
                result_acc[(nint, nstep, k)] = [(cacc, n)]
                MAEs[(nint, nstep, k)] = [(MAE, n)]
                seen.append((nint, nstep, k))

    cestimated_vars = dict_to_list(cestimated_vars)
    result_truevar = dict_to_list(result_truevar)
    result_acc = dict_to_list(result_acc)
    MAEs = dict_to_list(MAEs)


    return {"accuracy":result_acc,"MAE":MAEs, "variance":result_truevar, "estimated_variance":cestimated_vars}


run_random_sampling()
print "\n\n"
strat = run_stratified_sampling()

collected_strat = {}


collected_strat["OPT_A2"] = collect(strat[0])
collected_strat["OPT_A3"] = collect(strat[1])
collected_strat["OPT_A4"] = collect(strat[2])
collected_strat["OPT_A4_"] = collect(strat[3])
collected_strat["OPTIMAL"] = collect(strat[4])
collected_strat["EQUAL"] = collect(strat[5])
collected_strat["OPS_OPTIMAL"] = collect(strat[6])



pickle.dump(collected_strat, open(stratified_outfile+".data", "w"))
