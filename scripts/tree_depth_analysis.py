import sys
sys.path.append('.')

import json

from gridloc.gridloc_probe.analysis import TreeDepthAnalysis

tda = TreeDepthAnalysis(
        f'experiments/sent_eval_tasks/tree_depth.json',
        device=0)
# for seed in range(20):
seed = 0
with open(f'full_probe/tree_depth/{seed}/best.json') as best_epoch_file:
    best_epoch_json = json.load(best_epoch_file)
best_epoch = best_epoch_json['best_epoch']['epoch']
tda.set_seed(seed)
tda.load_checkpoint(f'gridloc_checkpoints/tree_depth/{seed}/epoch_{best_epoch}.pt')
tda.analyse_tree_depth()