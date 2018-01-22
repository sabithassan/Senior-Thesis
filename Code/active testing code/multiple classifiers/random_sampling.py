import random
import numpy as np


## Runs simple random sampling.
## nTotals: array of number of available samples,
## X: how many times experiment should be repeated
## test_matrix: Instances
## test_tl: true lables for the instances
def main(nTotals, X, model, test_matrix, test_tl):

    ## dictionary for storing results
    allResutls = {}
    allResutls["accuracy"] = []
    allResutls["variance"] = []
    allResutls["estimated_variance"] = []
    allResutls["MAE"] = []


    ## calculates true accuracy
    allPreds = model.predict(test_matrix)
    true_accuracy = np.mean(map(lambda x,y: x==y, allPreds, test_tl))
    print "the true Accuracy is:        ", true_accuracy

    ## run experiment for each value in nTotals
    for nTotal in nTotals:

        print "Random Sampling with n :", nTotal
        accuracies = []
        variances = []
        estimated_variances = []

        ## repeat experiment for X times
        for x in range (X):
            length = test_matrix.shape[0]
            indices = np.array(random.sample(range(0,length), nTotal))

            ## select nTotal random samples from the sparse matrix
            instances_value = test_matrix[indices]
            instances_tl = test_tl[indices]

            ## calculates accuracy of the model on these samples
            values = model.predict(instances_value)
            accuracy = np.mean(map(lambda x,y: x==y, values, instances_tl))

            ## estimated variance from formula
            estimated_variance = (1 / (nTotal - 1.0)) * (accuracy * (1 - accuracy)) #eq 3
            accuracies.append(accuracy)
            estimated_variances.append(estimated_variance)

        accuracy = np.mean(np.array(accuracies)) ## mean accuracy over X experiments
        variance = np.var(np.array(accuracies)) ## variance in the results
        estimated_variance = np.mean(np.array(estimated_variances)) ## estimated variance from the formula
        MAE = sum(map(lambda x: abs(x - true_accuracy), accuracies))/X ## Mean Average Error error

        ## accumulate results
        allResutls["accuracy"].append(accuracy)
        allResutls["variance"].append(variance)
        allResutls["estimated_variance"].append(estimated_variance)
        allResutls["MAE"].append(MAE)

        print "Accuracy               :", accuracy
        print "Variance               :", variance
        print "Estimated Variance     :", estimated_variance
        print "MAE                    :", MAE
        print "________________________"


    print allResutls

    return allResutls
