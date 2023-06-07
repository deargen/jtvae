import torch

all_smiles = []
for i in range(1,11):
    for j in range(5):
        fn = 'results%d/scores%d.pt' % (i,j)
        scores = torch.load(fn)
        fn = 'results%d/valid_smiles%d.pt' % (i,j)
        smiles = torch.load(fn)
        all_smiles.extend(list(zip(smiles, scores)))

all_smiles = [(x,-y) for x,y in all_smiles]
all_smiles = sorted(all_smiles, key=lambda x:x[1], reverse=True)
for s,v in all_smiles:
    print(s,v)
#mols = [Chem.MolFromSmiles(s) for s,_ in all_smiles[:50]]
#vals = ["%.2f" % y for _,y in all_smiles[:50]]
#img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200,135), legends=vals, useSVG=True)
#print img
