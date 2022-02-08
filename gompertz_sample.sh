
################################  Gompertz ODE Sample ################################

ode=GompertzODE
ode_param=1.5,1.5

# change sample size

seed_arr=( 0 50 )
noise=0.02
n_seed=50
sample_arr=( 5 20 100 )


for seed in "${seed_arr[@]}"
do
    for sample in "${sample_arr[@]}"
    do
         python -u run_simulation_vi.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed}
         python -u run_simulation.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed}
         python -u run_simulation.py --alg=spline --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed}
         python -u run_simulation.py --alg=gp --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed}
    done
done



## summarize sample

freq=10
sample_arr=( 5 20 50 100 )

rm results/GompertzODE-n.txt

for sample in "${sample_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE-n.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/GompertzODE-n.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/GompertzODE-n.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/GompertzODE-n.txt
done

cat results/GompertzODE-n.txt
