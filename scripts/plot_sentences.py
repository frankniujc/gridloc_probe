import sys
sys.path.append('.')
import json

from gridloc.gridloc_probe import GridLocProbeExperimentAnalysis

NUM_SENTENCE = 100

for task in ['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out', 'past_present', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content']:
    seed = 0
    with open(f'gridloc_checkpoints/{task}/{seed}/best.json') as best_epoch_file:
        best_epoch_json = json.load(best_epoch_file)
        best_epoch = best_epoch_json['best_epoch']['epoch']
    gridloc = GridLocProbeExperimentAnalysis(
        f'experiments/sent_eval_tasks/{task}.json',
        device=0)
    gridloc.set_seed(seed)
    gridloc.load_checkpoint(f'gridloc_checkpoints/{task}/{seed}/epoch_{best_epoch}.pt')
    gridloc.plot_sentences(NUM_SENTENCE, plot_directory=f'plots/{task}/sentences/')