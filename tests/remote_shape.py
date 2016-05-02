import numpy as np
import sys
import os

if __name__ == "__main__":
    data_file = sys.argv[1]
    data = np.load(data_file)
    print data.shape
    for v in data[:]:
        print v.shape

    os.unlink(data_file)