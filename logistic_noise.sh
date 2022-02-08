
################################  Logistic ODE  ################################

ode=LogisticODE

# change noise level

seed_arr=( 0 50 )

noise_arr=( 0.01 0.02 0.03 0.05 0.07 0.09 0.1 0.15 0.2 0.25 0.3 0.5 0.7 0.9 1.1 1.3 )

n_seed=50
freq=5

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        python -u run_simulation.py --alg=gp --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        python -u run_simulation.py --alg=spline --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        python -u run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
    done
done


## summarize noise

sample=50

rm results/LogisticODE-noise.txt

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/LogisticODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/LogisticODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/LogisticODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/LogisticODE-noise.txt
done

cat results/LogisticODE-noise.txt


