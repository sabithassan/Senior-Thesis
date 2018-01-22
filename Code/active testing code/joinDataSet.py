from sklearn.datasets import load_svmlight_file
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import random
import pickle
import numpy as np

X_train, Y_train = load_svmlight_file("../sshaar/datasets/epsilon/epsilon_normalized.bz2")
X_test, Y_test = load_svmlight_file("../sshaar/datasets/epsilon/epsilon_normalized.t.bz2")

allInstances = X_train + Y_train
allLabels = X_test + Y_test

print (allInstances.shape)
print (allLabels.shape)


# randomly pick 4000 training instances
trainIndices = np.array(random.sample(range(0,490000), 50000))
allIndices = np.array([i for i in range (490000)])
##
## the rest are testing instances
testIndices = np.setdiff1d(allIndices, trainIndices)
##
##print ("testing number", testIndices.shape)
##              
#### loads training instances
##trainInstances = X_train[trainIndices]
##trainLabels = Y_train[trainIndices]
##
#### loads testing instances
##testInstances = X_train[testIndices]
##testLabels = Y_train[testIndices]
##
#### dumps the files
pickle.dump( testIndices, open( "testIndices.pkl", "wb" ) )
print ("dumped indices")
##pickle.dump( testInstances, open( "testInstances.pkl", "wb" ) )
##pickle.dump( testLabels, open( "testLabels.pkl", "wb" ) )
##
#print (trainInstances.shape)


print ("training svm")
clf = LinearSVC()#multi_class="crammer_singer")#probability=True)#, cache_size=2000)
clf.fit(trainInstances, trainLabels)
print ("dumping svm")
joblib.dump(clf, 'svm1.pkl')
print ("svm dumped")

## loads the svm and computes accuracy
clf = joblib.load('svm1.pkl')
accuracy = clf.score(testInstances, testLabels)
print ("svm accuracy : ", accuracy)
