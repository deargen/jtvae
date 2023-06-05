import torch
import torch.nn as nn
from torch.autograd import Variable
from argparse import ArgumentParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
from sascorer import SAScorer

from tqdm.auto import tqdm
import numpy as np
from mol_gen.models.JT_VAE.jtnn import *
import networkx as nx

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = ArgumentParser()
parser.add_argument("-a", "--data", dest="data_path", required=True)
parser.add_argument("-v", "--vocab", dest="vocab_path", required=True)
parser.add_argument("-m", "--model", dest="model_path", required=True)
parser.add_argument("-s", "--sascorer", dest="sascorer_path", required=True)
parser.add_argument("-w", "--hidden", dest="hidden_size", default=200)
parser.add_argument("-l", "--latent", dest="latent_size", default=56)
parser.add_argument("-d", "--depth", dest="depth", default=3)
opts = parser.parse_args()

with open(opts.data_path) as f:
    smiles = f.readlines()

for i in range(len(smiles)):
    smiles[i] = smiles[i].strip()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

batch_size = 100
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

smiles_rdkit = []
for i in tqdm(range(len(smiles)), desc="Converting SMILES to RDKit"):
    smiles_rdkit.append(MolToSmiles(MolFromSmiles(smiles[i]), isomericSmiles=True))

logP_values = []
for i in tqdm(range(len(smiles)), desc="Calculating logP values"):
    logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[i])))

sascorer = SAScorer(opts.sascorer_path)
SA_scores = []
for i in tqdm(range(len(smiles)), desc="Calculating SA scores"):
    SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i])))

cycle_scores = []
for i in tqdm(range(len(smiles)), desc="Calculating cycle scores"):
    cycle_list = nx.cycle_basis(
        nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[i])))
    )
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_scores.append(-cycle_length)

SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(
    logP_values
)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(
    cycle_scores
)

latent_points = []
for i in tqdm(range(0, len(smiles), batch_size), desc="Calculating latent points"):
    batch = smiles[i : i + batch_size]
    mol_vec = model.encode_latent_mean(batch)
    latent_points.append(mol_vec.data.cpu().numpy())

# We store the results
latent_points = np.vstack(latent_points)
np.savetxt("latent_features.txt", latent_points)
torch.save(latent_points, "latent_features.pt")

targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
np.savetxt("targets.txt", targets)
torch.save(targets, "targets.pt")
np.savetxt("logP_values.txt", np.array(logP_values))
torch.save(logP_values, "logP_values.pt")
np.savetxt("SA_scores.txt", np.array(SA_scores))
torch.save(SA_scores, "SA_scores.pt")
np.savetxt("cycle_scores.txt", np.array(cycle_scores))
torch.save(cycle_scores, "cycle_scores.pt")
