from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import random
import pickle
import numpy as np

trainIndices = np.array(random.sample(range(0,20000), 50))
              
## loads training instances
trainInstances, trainLabels = load_svmlight_file("../datasets/rcv1/rcv1_train.binary.bz2")

trainInstances = trainInstances[trainIndices]
trainLabels = trainLabels[trainIndices]

## loads testing instances
testInstances, testLabels = load_svmlight_file("../datasets/rcv1/rcv1_test.binary.bz2")


print ("training svm")
clf = LinearSVC()#multi_class="crammer_singer")#probability=True)#, cache_size=2000)
clf.fit(trainInstances, trainLabels)
print ("dumping svm")
joblib.dump(clf, 'svm2.pkl')
print ("svm dumped")

## loads the svm and computes accuracy
clf = joblib.load('svm2.pkl')
accuracy = clf.score(testInstances, testLabels)
print ("svm accuracy : ", accuracy)
