
################################  Gompertz ODE Noise ################################

mkdir -p results/GompertzODE
mkdir -p results_vi/GompertzODE
mkdir -p results_spline/GompertzODE
mkdir -p results_gp/GompertzODE


ode=GompertzODE
ode_param=1.5,1.5


# change noise level

seed_arr=( 0 50 )

noise_arr=( 0.01 0.03 0.05 0.07 0.09 0.1 0.15 0.2 0.25 0.3 0.5 0.7 0.9 1.1 1.3 )
n_seed=50

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        python -u run_simulation.py --alg=spline --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        python -u run_simulation.py --alg=gp --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        python -u run_simulation_vi.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        sleep 1
    done
done


## summarize noise

freq=10
noise_arr=( 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.15 0.2 0.25 0.3 0.5 0.7 0.9 1.1 1.3 )
sample=50


rm results/GompertzODE-noise.txt

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/GompertzODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/GompertzODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/GompertzODE-noise.txt
    sleep 1
done

cat results/GompertzODE-noise.txt

