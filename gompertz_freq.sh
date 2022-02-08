
################################  Gompertz ODE  Freq ################################

ode=GompertzODE
ode_param=1.5,1.5

# change sampling frequency

seed_arr=( 0 50 )
noise=0.02
n_seed=50
sample=50
freq_arr=( 10 5 2 1 0.8 0.5 )


for seed in "${seed_arr[@]}"
do
    for freq in "${freq_arr[@]}"
    do
         python -u run_simulation.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
         python -u run_simulation_vi.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
         python -u run_simulation.py --alg=spline --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
         python -u run_simulation.py --alg=gp --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
    done
done


## summarize frequency

freq_arr=( 10 5 2 1 0.8 0.5 )

rm results/GompertzODE-freq.txt

for freq in "${freq_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/GompertzODE-freq.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/GompertzODE-freq.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/GompertzODE-freq.txt
    python -u evaluation.py --ode_name=${ode} --ode_param=${ode_param} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/GompertzODE-freq.txt
done

cat results/GompertzODE-freq.txt
