import random

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
