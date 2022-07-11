import argparse
import os
import pickle


def list_average(a: list, rnd: bool = True):
    avg = sum(a) / len(a)
    avg = avg * 100
    if rnd:
        avg = round(avg, 1)
    
    return avg


"""
>>> split_list([1,2,3,4],3)
[[1, 2], [3], [4]]
"""
def split_list(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def kth_highest_acc(k: int, lst: list, rnd: bool = True):
    k = min(k, len(lst)-2)
    k = max(0, k)

    if k == len(lst)-1 and k == 0:
        k = -1

    res = sorted(lst, reverse=True)[k+1]
    res = res * 100

    if rnd:
        res = round(res, 1)
    
    return res
    

def main():
    parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-metrics", type=str.lower, choices = ["average", "kthlargest", "all"], 
                        default="all", help="choose what metrics to calculate")
    parser.add_argument("-k", help="return kth largest val acc obtained", type=int, default=2)
    parser.add_argument("-intervals", help="number of intervals to display for val accs", 
                        type=int, default=5)
    parser.add_argument("-pfile", help="Training pickle file to analyse", type=str, required=True)
    
    args = vars(parser.parse_args())

    pickle_file_path = os.path.abspath(args["pfile"])
    
    f = open(pickle_file_path, 'rb')
    hist = pickle.load(f)
    f.close()
    
    val_acc_list = hist['val_accuracy']
    
    print(f"Average val accuracy = {list_average(val_acc_list)}")

    intervals = int(args["intervals"])
    interval_wise_val_accs = [list_average(sublist) 
                              for sublist in split_list(val_acc_list, intervals)  if sublist!=[]]
    
    print(f"Interval Wise Val Acc = {interval_wise_val_accs}")
    
    k = int(args["k"])
    l = len(val_acc_list)
    
    print(f"Peak val accuracy (ranked {k} out of {l} epochs) = {kth_highest_acc(k, val_acc_list)}")
    
    
    
if __name__ == "__main__":
    main()
    