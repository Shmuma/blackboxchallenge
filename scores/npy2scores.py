import json
import pprint
import sys
sys.path.append("..")

import argparse
from time import time

from lib import net_light, infra, run_bbox


def run_test(args, network_weights, test_mode):
    scores = {}

    def step_hook():
        step = infra.bbox.get_time()
        if step > 0 and step % args.ticks == 0:
            if not step in scores:
                scores[step] = []
            scores[step].append(infra.bbox.get_score())

    t = time()
    for _ in xrange(args.rounds):
        score, _ = run_bbox.test_performance_no_tf(network_weights,
                                             alpha=args.alpha, max_steps=args.steps,
                                             test_level=False, step_hook=step_hook)
        if not args.steps in scores:
            scores[args.steps] = []
        scores[args.steps].append(score)

    return {
        'scores': scores,
        'time': time() - t
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Amount of rounds to perform, default=10")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha value for testing, default=0.05")
    parser.add_argument("--steps", type=int, default=100000, help="Limit amount of steps, default=100k")
    parser.add_argument("--ticks", type=int, default=50000, help="Measure scores every ticks steps, default=50k")
    parser.add_argument("npys", nargs="+", help="NPY model files to process")
    args = parser.parse_args()

    infra.prepare_bbox()

    for npy in args.npys:
        print "Process file %s" % npy

        network_weights = net_light.load_weights(npy)
        res = run_test(args, network_weights, test_mode=False)
        pprint.pprint(res)



