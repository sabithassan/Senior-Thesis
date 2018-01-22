# from sklearn.cluster import KMeans
import numpy as np
import gzip
import pickle
import random
import files_reader as reader


def run(nTotal, data):
    true_vs_class_labels = data["true_vs_class_labels"]
    samples = random.sample(true_vs_class_labels, nTotal)
    accuracy = (reduce(lambda x, y: x + y, samples) / float(nTotal))
    varience = (1 / (nTotal - 1.0)) * (accuracy * (1 - accuracy))
    return (accuracy, varience)

def main(data, X, nTotals):
    allResutls = ([], [], [], [])
    print len(data["true_vs_class_labels"])
    trueaccs = sum(data["true_vs_class_labels"])/float(len(data["true_vs_class_labels"]))

    for nTotal in nTotals:
        acc_var = []
        for x in range(X):
            (ac, var) = run(nTotal,data)
            acc_var.append((ac, var))
        # print acc_var
        accs = map(lambda (x, y): x, acc_var)
        varss = map(lambda (x, y): y, acc_var)
        avg_acc = reduce(lambda x, y: x + y, accs)/float(len(accs))
        avg_var = reduce(lambda x, y: x + y, varss)/float(len(accs))
        true_var = np.var(np.array(accs))
        allResutls[0].append(avg_acc)
        allResutls[1].append(avg_var)
        allResutls[2].append(true_var)

        allResutls[3].append(reduce(lambda x , y: x + y, map(lambda x: abs(x - trueaccs), accs))/X)
    return allResutls

# results = main()
# print results
#
# plt.plot(results[0], nTotals)
# plt.plot(results[0], nTotals)
# plt.show()
