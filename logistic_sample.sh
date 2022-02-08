
################################  Logistic ODE  ################################

ode=LogisticODE



# change sample size

seed_arr=( 0 50 )
noise=0.15
n_seed=50
sample_arr=( 5 20 100 )
freq=5

for seed in "${seed_arr[@]}"
do
    for sample in "${sample_arr[@]}"
    do
        python -u run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed} --freq=${freq}
        python -u run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed} --freq=${freq}
        python -u run_simulation.py --alg=spline --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed} --freq=${freq}
        python -u run_simulation.py --alg=gp --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_sample=${sample} --n_seed=${n_seed} --freq=${freq}
    done
done


## summarize sample

sample_arr=( 5 20 50 100 )

rm results/LogisticODE-n.txt

for sample in "${sample_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/LogisticODE-n.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/LogisticODE-n.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/LogisticODE-n.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/LogisticODE-n.txt
done

cat results/LogisticODE-n.txt
