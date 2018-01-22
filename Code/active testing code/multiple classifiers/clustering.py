from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LogisticRegression



## runs KMEANS on classifier scores of different models
def KMeanss(models, K, test_matrix, test_tl):

    allScores = []
    for model in models:
        print ("getting probabilities")
        ## will get the scores of model on the test dataset
        probabilities = model.predict_proba(test_matrix)

        print ("getting max probability")
        ## picks the max between different probabilities
        scores = np.array(list (map (lambda x: max(x), probabilities)))

        allScores.append(scores)

    ## transposes allScores. creates a 2D matrix where each vector contains
    ## all classifier scores on that instance
    allScores = np.array(allScores)
    print ("shape of KMEANS input", allScores.shape)
            
    print ("Running KMeans")
    km = KMeans(n_clusters = K, max_iter=20).fit(scores)

    clusters = {}
    mapped = km.labels_

    print ("Mapping")

    for i in range(len(mapped)):
        c = mapped[i]
        if c not in clusters:
            clusters[c] = [(test_matrix[i], model.predict(test_matrix[i])[0], test_tl[i])]
        else:
            clusters[c].append((test_matrix[i], model.predict(test_matrix[i])[0], test_tl[i]))

    print ("Finished Mapping")

    return clusters
