import numpy as np
import argparse

def norm(v,p):
    arr = np.asarray(v, dtype=np.float32)
    return np.power(np.sum(np.power(abs(arr),int(p))),1/int(p))

parser = argparse.ArgumentParser(description='p norm')
parser.add_argument('numbers', type=float,nargs='+')
parser.add_argument('--p', default=2)
args = parser.parse_args()
x = norm(args.numbers,args.p)  
print("Norm of",args.numbers,"is","{:.2f}".format(x))               