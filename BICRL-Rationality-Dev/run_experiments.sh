#!/bin/sh


python3 main.py --N_traj=100  --boltz_beta_demo=1  --boltz_beta_infer=2 --noise=0 --cnstr_alloc=3
python3 main.py --N_traj=100  --boltz_beta_demo=1  --boltz_beta_infer=5 --noise=0 --cnstr_alloc=3
python3 main.py --N_traj=100  --boltz_beta_demo=1  --boltz_beta_infer=10 --noise=0 --cnstr_alloc=3
python3 main.py --N_traj=100  --boltz_beta_demo=1  --boltz_beta_infer=100 --noise=0 --cnstr_alloc=3
python3 main.py --N_traj=100  --boltz_beta_demo=1  --boltz_beta_infer=1000 --noise=0 --cnstr_alloc=3


