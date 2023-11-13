conda create -n ft_mcq python=3.10
conda activate ft_mcq

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install accelerate
pip install accelerate
conda install datasets transformers
