import sys
sys.path.append('.')

from gridloc.gridloc_probe import GridLocProbeExperiment

for task in ['bigram_shift', 'coordination_inversion', 'obj_number', 'odd_man_out', 'past_present', 'sentence_length', 'subj_number', 'top_constituents', 'tree_depth', 'word_content']:
    for i in range(20):
        gridloc = GridLocProbeExperiment(f'experiments/sent_eval_tasks/{task}.json', device=0)
        assert gridloc.config.bert_version == 'bert-base-uncased'
        assert gridloc.config.output_directory.startswith('output/')
        gridloc.set_seed(i)
        gridloc.config.output_directory += f'/{i}'
        gridloc.create_save_directory()
        gridloc.probe()