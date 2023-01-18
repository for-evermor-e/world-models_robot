https://github.com/ctallec/world-models

conda create -n python=3.7

pip3 install -r requirements.txt

mjpro150 (not mujoco210)

pip3 install torch
pip3 install torchvision

pip3 install gym==0.15.3 (not gym 0.26.2)

python data/generation_script.py --rollouts 100 --rootdir datasets/carracing --threads 8

python trainvae.py --logdir exp_dir
