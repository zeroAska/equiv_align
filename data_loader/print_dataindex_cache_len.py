import pickle
import sys


with open(sys.argv[1], 'rb') as f:
    a = pickle.load(f)
    print(len(a[0]))

