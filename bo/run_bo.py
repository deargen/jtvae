import pickle
import gzip
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
import os
import os.path

import rdkit
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors

import torch
import torch.nn as nn
from mol_gen.models.JT_VAE.jtnn import create_var, JTNNVAE, Vocab

from argparse import ArgumentParser

from sascorer import SAScorer
import networkx as nx
from rdkit.Chem import rdmolops

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = ArgumentParser()
parser.add_argument("-v", "--vocab", dest="vocab_path", required=True)
parser.add_argument("-m", "--model", dest="model_path", required=True)
parser.add_argument("-o", "--save_dir", dest="save_dir", required=True)
parser.add_argument("-f", "--features", dest="features_path", required=True)
parser.add_argument("-s", "--sascorer", dest="sascorer_path", required=True)
parser.add_argument("-w", "--hidden", dest="hidden_size", default=200)
parser.add_argument("-l", "--latent", dest="latent_size", default=56)
parser.add_argument("-d", "--depth", dest="depth", default=3)
parser.add_argument("-r", "--seed", dest="random_seed", default=None)
parser.add_argument("-i", "--iteration", dest="iteration", default=5)
parser.add_argument("-n", "--num_per_iter", dest="num_per_iter", default=60)
opts = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
random_seed = int(opts.random_seed)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

# initialize the SA scorer
sascorer = SAScorer(opts.sascorer_path)

# We load the random seed
np.random.seed(random_seed)

# We load the data (y is minued!)
latent_label_dict = torch.load(opts.features_path)
X = latent_label_dict['latent_points']
y = - latent_label_dict['targets']
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

np.random.seed(random_seed)

logP_values = latent_label_dict['logP_values']
SA_scores = latent_label_dict['SA_scores']
cycle_scores = latent_label_dict['cycle_scores']
SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

# save directory
os.makedirs(opts.save_dir, exist_ok = True)

iteration = 0
while iteration < opts.iteration:
    # We fit the GP
    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)

    # We pick the next 60 inputs
    next_inputs = sgp.batched_greedy_ei(opts.num_per_iter, np.min(X_train, 0), np.max(X_train, 0))
    valid_smiles = []
    new_features = []
    for i in range(opts.num_per_iter):
        all_vec = next_inputs[i].reshape((1,-1))
        tree_vec,mol_vec = np.hsplit(all_vec, 2)
        tree_vec = create_var(torch.from_numpy(tree_vec).float())
        mol_vec = create_var(torch.from_numpy(mol_vec).float())
        s = model.decode(tree_vec, mol_vec, prob_decode=False)
        if s is not None: 
            valid_smiles.append(s)
            new_features.append(all_vec)
    
    print(len(valid_smiles), "molecules are found")
    new_features = np.vstack(new_features)
    torch.save(valid_smiles, opts.save_dir + "/valid_smiles{}.pt".format(iteration))

    scores = []
    for i in range(len(valid_smiles)):
        current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles[ i ]))
        current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles[ i ]))
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles[ i ]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6

        current_cycle_score = -cycle_length
     
        current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
        current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
        current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

        score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
        scores.append(-score) #target is always minused
    torch.save(scores, opts.save_dir + "/scores{}.pt".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1