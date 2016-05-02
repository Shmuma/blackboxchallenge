from fabric.api import execute, env, run
from fabric.operations import put

import numpy as np
import os

PATH = "/tmp/test.data.npy"

def uname():
    data1 = np.zeros((10,))
    data2 = np.zeros((100,))
    np.save(PATH, [data1, data2])
    put(PATH, PATH)
    res = run("python remote_shape.py %s" % PATH)
    os.unlink(PATH)
    return res

if __name__ == "__main__":
    env.use_ssh_config = True

    print execute(uname, hosts=['test'])