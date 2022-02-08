
################################  Frac ODE  ################################

ode=FracODE

# change noise level

seed_arr=( 0 )

noise_arr=( 0.01 0.1 0.3 0.5 0.7 0.9 )
n_seed=50
freq=100

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        python -u run_simulation.py --alg=spline --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        python -u run_simulation.py --alg=gp --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
        python -u run_simulation_vi.py --ode_name=${ode} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --freq=${freq} --n_sample=50
    done
done


noise_arr=( 0.01 0.1 0.3 0.5 0.7 0.9 )
sample=50
freq=100

ode=FracODE

rm results/FracODE-noise.txt

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/FracODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/FracODE-noise.txt 
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/FracODE-noise.txt 
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/FracODE-noise.txt 
done

cat results/FracODE-noise.txt


