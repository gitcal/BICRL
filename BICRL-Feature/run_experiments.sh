#!/bin/sh


# # 
python3 main2.py --N_demonstrations=20 --num_steps=2000 --iterations=50 --method='BICRL' #--noise=0.025
python3 main2.py --N_demonstrations=20 --num_steps=2000 --iterations=50 --method='pen_rew_disc' #--noise=0.025
python3 main2.py --N_demonstrations=20 --num_steps=2000 --iterations=50 --method='pen_rew_cont' #--noise=0.025
python3 main2.py --N_demonstrations=20 --num_steps=2000 --iterations=50 --method='BIRL' #--noise=0.025

