
################################  Selkov ODE  ################################

ode=SelkovODE

# change noise level

seed_arr=( 0 50 )
n_seed=50

noise_arr=( 0.01 0.1 0.2 )
freq=10

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


## summarize noise

sample=50

rm results/SelkovODE-noise.txt

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/SelkovODE-noise.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/SelkovODE-noise.txt 
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/SelkovODE-noise.txt 
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/SelkovODE-noise.txt 
done

cat results/SelkovODE-noise.txt

## parameter accuracy

rm results/SelkovODE-noise-param.txt

for noise in "${noise_arr[@]}"
do
    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=vi >> results/SelkovODE-noise-param.txt
    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=diff >> results/SelkovODE-noise-param.txt 
    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=spline >> results/SelkovODE-noise-param.txt 
    python -u evaluation_param_selkov.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=gp >> results/SelkovODE-noise-param.txt 
done

cat results/SelkovODE-noise-param.txt


