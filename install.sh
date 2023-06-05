#conda environment
# conda create -n jt-vae python=3.6 rdkit=2017.09 -c conda-forge

#install pytorch
conda install -c pytorch pytorch=1.10.2=py3.6_cuda11.3_cudnn8.2.0_0 -y
pip install -r requirements.txt