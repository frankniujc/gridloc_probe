import sys
sys.path.append('.')

from gridloc.gridloc_probe import GridLocProbeExperimentAnalysis

for task in ['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out', 'past_present', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content']:
    gridloc = GridLocProbeExperimentAnalysis(
        f'experiments/sent_eval_tasks/{task}.json',
        device=0)
    for seed in range(20):
        for epoch in range(30):
            gridloc.set_seed(seed)
            gridloc.load_checkpoint(f'gridloc_checkpoints/{task}/{seed}/epoch_{epoch}.pt')
            gridloc.layer_weights_center('test', plot_directory=f'plots/{task}/{seed}/', name=f'layer_weight_center_epoch_{epoch}')