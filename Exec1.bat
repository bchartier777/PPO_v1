Rem Batch execution of PPO

python main.py --train_test train --act_file "" --crit_file "" --max_tstep 1000000 --tstep_batch 2048 --tstep_ep 200 --gamma 0.99 --env_name LunarLanderContinuous-v2 --lr 3e-4 --clip 0.2 --img_rend --img_rend_freq 10 --checkpoint 10 1>ExecOut\stdOutPPO_Lunarv2.txt  2>ExecOut\stdErrPPO_Lunarv2.txt

python main.py --train_test train --act_file "" --crit_file "" --max_tstep 1000000 --tstep_batch 2048 --tstep_ep 200 --gamma 0.99 --env_name BipedalWalker-v3 --lr 3e-4 --clip 0.2 --img_rend --img_rend_freq 10 --checkpoint 10 1>ExecOut\stdOutPPO_Bipdeal.txt  2>ExecOut\stdErrPPO_Bipedal.txt

python main.py --train_test train --act_file "" --crit_file "" --max_tstep 1000000 --tstep_batch 2048 --tstep_ep 200 --gamma 0.99 --env_name Pendulum-v0 --lr 3e-4 --clip 0.2 --img_rend --img_rend_freq 10 --checkpoint 10 1>ExecOut\stdOutPPO_Pendulum.txt  2>ExecOut\stdErrPPO_Pendulum.txt

python main.py --train_test train --act_file "" --crit_file "" --max_tstep 1000000 --tstep_batch 2048 --tstep_ep 200 --gamma 0.99 --env_name MountainCarContinuous-v0 --lr 3e-4 --clip 0.2 --img_rend --img_rend_freq 10 --checkpoint 10 1>ExecOut\stdOutPPO_MountainCar.txt  2>ExecOut\stdErrPPO_MountainCar.txt

Rem "Execution complete"
