"""
Normalisations
"""
import infra

import json
import numpy as np
import logging as log


class Normaliser:
    def __init__(self, data):
        self.data = data


class NormEstimator:
    def __init__(self):
        self.sum = np.zeros(infra.n_features)
        self.m = np.zeros(infra.n_features)
        self.s = np.zeros(infra.n_features)
        self.count = 0

    def process(self, state):
        self.count += 1

        tmpM = np.array(self.m)
        self.m += (state - tmpM) / self.count
        self.s += (state - tmpM) * (state - self.m)
        self.sum += state

    def toData(self):
        return {
            'mean': (self.sum / self.count).tolist(),
            'std': np.sqrt(self.s / (self.count-2)).tolist()
        }

    def toString(self):
        return json.dumps(self.toData())


# Data is already normalized
norm_json = """
{"std": [0.9999999253536043, 1.000000477632064, 1.0000000583758901, 0.9999990311820047, 0.9999986536631041, 0.9999999957305914,
1.0000000225040193, 0.9999999207781298, 0.9999999344160696, 1.000000057142784, 1.000000075741048, 1.0000015411917733,
0.9999998736135233, 1.000003280268513, 1.000001719114416, 1.0000001570663326, 1.000006807714634, 0.9999998491880765,
1.0000017113841382, 0.999999863371927, 0.9999998062077516, 0.9999994001866799, 1.0000000102822135, 0.9999997649863065,
0.9999995258970028, 1.0000002939333217, 0.9999997583461698, 0.9999998175467976, 0.9999997508802226, 0.9999997101932265,
1.0000003316400214, 0.9999997367083501, 0.9999989092321344, 0.9999994209834936, 0.9999994944063455, 0.6737210369915366],

"mean": [-4.380888436088418e-07, 2.6626897499965935e-06, -6.591321105479383e-07, 1.1487075822738804e-06, 1.376831367493442e-06,
-5.365519825492016e-07, -5.629045801248296e-07, -5.027192518968677e-07, -4.826331189376875e-07, -7.066701721658582e-07,
-6.63646104984801e-07, -1.668144445443394e-06, -4.005034327988544e-07, -2.8322398502495233e-06, -3.4515725819405505e-06,
-1.996573320299173e-06, -4.224399083362405e-06, -2.8078100127708754e-07, -3.0215846442015936e-06, -6.595394918277716e-07,
-7.06968837173465e-08, -5.737985102777099e-07, 1.7929902793793293e-06, 3.181611157283889e-07, -5.060197906371233e-07,
-1.6154496977494552e-06, -2.388396070547392e-07, 6.874025054357811e-07, 8.952447347057215e-08, 5.631997242824477e-07,
1.0925610883454425e-06, 2.3589633487590976e-07, -4.148250152510279e-07, 1.3873991092175641e-06, -9.941495656418837e-07,
0.10055018840140814]}
"""


if __name__ == "__main__":
    infra.setup_logging()
    infra.prepare_bbox()
    log.info("Estimate normalisation vector for state")

    state = {
        'est': NormEstimator()
    }

    def action_reward_hook(our_state, bbox_state, rewards, next_states):
        for state in next_states:
            our_state['est'].process(state)
        return np.argmax(rewards)

    infra.bbox_checkpoints_loop(state, action_reward_hook, verbose=100000)
    log.info("Processed {count} states, score={score}".format(
            count=state['est'].count, score=infra.bbox.get_score()))
    print state['est'].toString()
