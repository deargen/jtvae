import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable
from optparse import OptionParser

import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops

from tqdm.auto import tqdm
import numpy as np
from mol_gen.models.JT_VAE.jtnn import *
from mol_gen.models.JT_VAE.bo.sascorer import SAScorer
from numpy.testing import assert_array_equal

class TestGenLatent(unittest.TestCase):
    def setUp(self):
        data_path = "/aicenter2/mol_generation/ckpts/JT_VAE/data/zinc/train.txt"
        vocab_path = "/aicenter2/mol_generation/ckpts/JT_VAE/data/zinc/vocab.txt"
        model_path = "/aicenter2/mol_generation/ckpts/JT_VAE/molvae/MPNVAE-h450-L56-d3-beta0.005/model.iter-4"
        sascorer_path = "/aicenter2/mol_generation/ckpts/JT_VAE/data/fpscores.pkl.gz"
        self.sascorer = SAScorer(sascorer_path)
        with open(data_path) as f:
            smiles = f.readlines()
        for i in range(len(smiles)):
            smiles[i] = smiles[i].strip()
        self.smiles = smiles

        vocab = [x.strip("\r\n ") for x in open(vocab_path)]
        vocab = Vocab(vocab)

        self.batch_size = 100

        model = JTNNVAE(vocab, hidden_size=450, latent_size=56, depth=3)
        model.load_state_dict(torch.load(model_path))
        self.model = model.cuda()

    def test_cal_values(self):
        smiles_rdkit = []
        for i in range(len(self.smiles)):
            smiles_rdkit.append(
                MolToSmiles(MolFromSmiles(self.smiles[i]), isomericSmiles=True)
            )

        logP_values = []
        for i in range(len(self.smiles)):
            logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[i])))

        SA_scores = []
        for i in range(len(self.smiles)):
            SA_scores.append(
                -self.sascorer.calculateScore(MolFromSmiles(smiles_rdkit[i]))
            )
    
    def test_latent_points(self):
        latent_features = np.loadtxt("latent_features.txt")
        latent_features_torch = torch.load("latent_features.pt")
        assert_array_equal(latent_features, latent_features_torch)