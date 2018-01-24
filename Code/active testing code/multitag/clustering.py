from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression



## runs KMEANS on classifier scores of different models
def KMeanss(models, K, test_matrix, test_tl):

    ## transposes test labels so each row has all corresponding true labels
    test_tl = test_tl.T

    allScores = []
    allPreds = []
    count = 1
    for model in models:
        
        print ("getting model,", count)
        ## will get the scores and predicted labelsof model on the test dataset
        probabilities = model.predict_proba(test_matrix)
        predictions = model.predict(test_matrix)
        #print ("predictions shape", predictions.shape)

        #print ("getting max probability")
        ## picks the max between different probabilities
        scores = np.array(list (map (lambda x: max(x), probabilities)))

        ## accumulates scores and predictions
        allScores.append(scores)
        allPreds.append(predictions)

        count += 1

    ## transposes allScores. creates a 2D matrix where each vector contains
    ## all classifier scores on that instance
    allScores = np.array(allScores).T
    allPreds = np.array(allPreds).T
    
    print ("shape of KMEANS input", allScores.shape)
    print ("shape of predictions", allPreds.shape)

            
    print ("Running KMeans")
    km = KMeans(n_clusters = K, max_iter=20).fit(allScores)

    clusters = {}
    mapped = km.labels_

    print ("Mapping")

    for i in range(len(mapped)):
        c = mapped[i]
        if c not in clusters:
            clusters[c] = [(test_matrix[i], allPreds[i], test_tl[i])]
        else:
            clusters[c].append((test_matrix[i], allPreds[i], test_tl[i]))

    print ("Finished Mapping")

    return clusters
