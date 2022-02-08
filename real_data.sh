
################################  Real Data  ################################

# diff

seed_arr=( 0 )

sample_arr=( 200 )

n_seed=50

for seed in "${seed_arr[@]}"
do
    for sample in "${sample_arr[@]}"
    do
        python -u run_simulation_real.py --seed=${seed} --n_seed=${n_seed} --n_sample=${sample}
    done
done

# vi

n_seed=40

for seed in "${seed_arr[@]}"
do
    for sample in "${sample_arr[@]}"
    do
        python -u run_simulation_real_vi.py --seed=${seed} --n_seed=${n_seed} --n_sample=${sample}
    done
done
