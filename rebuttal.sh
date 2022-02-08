
################################  Gompertz ODE  ################################

ode=GompertzODE
ode_param=1.5,1.5


# change noise level

seed_arr=( 50 )

noise_arr=( 0.01 0.1 0.3 0.5 0.7 0.9 1.1 1.3 )
n_seed=50

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation_node.py --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        python -u run_simulation_node.py --alg=one-step --ode_name=${ode} --ode_param=${ode_param} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        sleep 1
    done
done



freq=10
sample=50

ode=GompertzODE
ode_param=1.5,1.5


for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=node >> results/GompertzODE-noise.txt
        python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --alg=node_one_step >> results/GompertzODE-noise.txt
        sleep 1
    done
done

################################  Lorenz ODE  ################################

ode=Lorenz


# change noise level

seed_arr=( 0 )

noise_arr=( 0.09 0.15 0.2 0.25 0.3 )

n_seed=50
freq=25
x_id=0

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation_node.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
        python -u run_simulation_node.py --alg=one-step --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
    done
done


freq=25
noise_arr=( 0.09 0.15 0.2 0.25 0.3 )
sample=50
x_id=0
ode=Lorenz

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --x_id=${x_id} --alg=node >> results/Lorenz-noise.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --x_id=${x_id} --alg=node_one_step >> results/Lorenz-noise.txt 
    sleep 1
done

cat results/Lorenz-noise.txt


# Equation 2

ode=Lorenz

seed_arr=( 0 )

noise_arr=( 0.09 0.15 0.2 0.25 0.3 )

n_seed=50
freq=50
x_id=1

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation_node.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
        python -u run_simulation_node.py --alg=one-step --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
        sleep 1
    done
done




freq=50
noise_arr=( 0.09 0.15 0.2 0.25 0.3 )
sample=50
x_id=1
ode=Lorenz

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --x_id=${x_id} --alg=node >> results/Lorenz-noise-1.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --x_id=${x_id} --alg=node_one_step >> results/Lorenz-noise-1.txt 
    sleep 1
done

cat results/Lorenz-noise-1.txt



# Equation 3

ode=Lorenz

seed_arr=( 0 )

noise_arr=( 0.09 0.15 0.2 0.25 0.3 )

n_seed=50
freq=50
x_id=2

for seed in "${seed_arr[@]}"
do
    for noise in "${noise_arr[@]}"
    do
        python -u run_simulation_node.py --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
        python -u run_simulation_node.py --alg=one-step --ode_name=${ode} --seed=${seed} --freq=${freq} --noise_sigma=${noise} --n_seed=${n_seed} --x_id=${x_id} --n_sample=50
        sleep 1
    done
done



freq=50
noise_arr=( 0.09 0.15 0.2 0.25 0.3 )
sample=50
x_id=2
ode=Lorenz

for noise in "${noise_arr[@]}"
do
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --x_id=${x_id} --alg=node >> results/Lorenz-noise-2.txt
    python -u evaluation.py --ode_name=${ode} --noise_sigma=${noise} --n_sample=${sample} --freq=${freq} --x_id=${x_id} --alg=node_one_step >> results/Lorenz-noise-2.txt 
    sleep 1
done

cat results/Lorenz-noise-2.txt

