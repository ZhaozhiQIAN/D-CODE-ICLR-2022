
################################  Logistic ODE  ################################

ode=LogisticODE


# change sampling frequency

seed_arr=( 0 50 )
noise=0.15
n_seed=50
sample=50
freq_arr=( 10 5 2 1 0.8 0.5 )


for seed in "${seed_arr[@]}"
do
    for freq in "${freq_arr[@]}"
    do
        python -u run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
        python -u run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
        python -u run_simulation.py --alg=spline --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
        python -u run_simulation.py --alg=gp --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --n_seed=${n_seed}
    done
done


## summarize frequency

rm results/LogisticODE-freq.txt

for freq in "${freq_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/LogisticODE-freq.txt &
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/LogisticODE-freq.txt  &
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/LogisticODE-freq.txt  &
    sleep 1
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/LogisticODE-freq.txt  &
done

cat results/LogisticODE-freq.txt
