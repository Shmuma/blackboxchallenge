import sys
sys.path.append("..")

import argparse

from lib import net_light, infra


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Amount of rounds to perform, default=10")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for testing, default=0.05")
    parser.add_argument("npys", nargs="+", help="NPY model files to process")
    args = parser.parse_args()

    infra.prepare_bbox()

    for npy in args.npys:
        print "Process file %s" % npy

        network_weights = net_light.load_weights(npy)
        for name, arr in sorted(network_weights.iteritems()):
            print name, arr.shape




