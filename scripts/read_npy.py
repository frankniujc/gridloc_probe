import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file', help='path to the npy file')
args = parser.parse_args()

path = args.file

file = open(path, 'rb')

batch_layer_attention_weight = True
while True:
    try:
        batch_layer_attention_weight = np.load(file)
        print(batch_layer_attention_weight.shape)
    except ValueError:
        print(batch_layer_attention_weight)
        print('Reached end of the file')
        break