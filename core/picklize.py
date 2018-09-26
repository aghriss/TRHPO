import pickle
import gzip


def picklize(D,fname):
    with open(fname, 'wb') as f:
        pickle.dump(D,f)
        f.close()

def depicklize(fname):
    with open(fname, 'r') as f:
        data = pickle.load(f,encoding="latin1")
        f.close()
    return data

def gpicklize(D,fname):
    with gzip.open(fname, 'wb') as f:
        pickle.dump(D,f)
        f.close()

def gdepicklize(fname):
    with gzip.open(fname, 'r') as f:
        data = pickle.load(f,encoding="latin1")
        f.close()
    return data
