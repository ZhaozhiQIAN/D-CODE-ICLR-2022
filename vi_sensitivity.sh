############# Gompertz ODE #############

ode=GompertzODE
ode_param=1.5,1.5


# change number of basis function

basis_arr=( 5 10 30 50 70 100 )

seed=0
noise=0.9
n_seed=100

alg_array=( sine cubic )

for n_basis in "${basis_arr[@]}"
do
    for basis in "${alg_array[@]}"
    do
        python -u run_sensitivity_vi.py --basis=${basis} --ode_name=${ode} --ode_param=${ode_param} --n_basis=${n_basis} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50
        sleep 1
    done
done


#rm results/sensitivity_${ode}.txt

for n_basis in "${basis_arr[@]}"
do
    for basis in "${alg_array[@]}"
    do
        python -u evaluation_sensitivity.py --basis=${basis} --ode_name=${ode} --ode_param=${ode_param} --n_basis=${n_basis} --seed=${seed} --noise_sigma=${noise} --n_seed=${n_seed} --n_sample=50 >> results/sensitivity_${ode}.txt
    done
done

cat results/sensitivity_${ode}.txt


############# Lorenz ODE #############



ode=Lorenz
x_id=1
seed=0
noise=0.3
n_seed=50
freq=50.0

basis_arr=( 5 10 30 50 70 100 )


for n_basis in "${basis_arr[@]}"
do
        python -u run_sensitivity_vi.py --ode_name=${ode} --basis=cubic --n_basis=${n_basis} --seed=${seed} --freq=${freq}  --noise_sigma=${noise} --x_id=${x_id} --n_seed=${n_seed} --n_sample=50 > results/${ode}/noise-${noise}-seed-${seed}.txt
        sleep 1
done




alg_array=( sine cubic )
rm results/sensitivity_${ode}.txt

for n_basis in "${basis_arr[@]}"
do
    for basis in "${alg_array[@]}"
    do
        python -u evaluation_sensitivity.py --ode_name=${ode} --basis=${basis} --n_basis=${n_basis} --seed=${seed} --freq=${freq}  --noise_sigma=${noise} --x_id=${x_id} --n_seed=${n_seed} --n_sample=50 >> results/sensitivity_${ode}.txt
    done
done

cat results/sensitivity_${ode}.txt

