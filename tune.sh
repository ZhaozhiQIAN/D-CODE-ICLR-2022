#
#ode=SelkovODE
#python -u tune_gp.py  --ode_name=${ode} --alg=gp --itr=3 > results/${ode}/tune_hyper.txt &
#
#
#
#
#ode=SelkovODE
##seed_arr=( 0 1 2 3 4 5 6 7 8 9 )
#seed_arr=( 0 1 2 3 4 )
#
#for seed in "${seed_arr[@]}"
#do
#    nohup python -u tune_gp.py  --ode_name=${ode} --alg=gp --itr=10 --seed=${seed} > results/${ode}/tune_hyper_${seed}.txt &
#
#done
#
#
#ode=SelkovODE
#seed_arr=( 1 2 3 4 5 )
#
#for seed in "${seed_arr[@]}"
#do
#    python -u run_simulation.py --noise_sigma=0.1 --ode_name=${ode} --alg=gp --seed=${seed} > results/${ode}/noise-0.1-seed-${seed}.txt &
#done


################################  Gompertz ODE  ################################

# tuning the hyper-parameters

ode=GompertzODE
ode_param=1.5,2.718281828459045

seed_arr=( 0 1 2 3 4 )

mkdir results/${ode}
for seed in "${seed_arr[@]}"
do
    nohup python -u tune_gp.py  --ode_name=${ode} --ode_param=${ode_param} --alg=gp --itr=10 --const_max=-1.5 --const_min=-1.5 --n_sample=5 --seed=${seed}  > results/${ode}/tune_hyper2_${seed}.txt &
done

# reduce constant range, reduce sample size
# reprogram log function
# add -1 as constant





################################  Lorenz ODE  ################################

ode=Lorenz
x_id=0


seed_arr=( 0 1 2 3 4 )

mkdir results/${ode}
for seed in "${seed_arr[@]}"
do
    nohup python -u tune_gp.py  --ode_name=${ode} --x_id=${x_id} --const_max=30 --alg=gp --itr=10 --seed=${seed} > results/${ode}/tune_hyper_x_${x_id}_${seed}.txt &
done



