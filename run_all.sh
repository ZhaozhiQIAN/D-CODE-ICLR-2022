#!/bin/bash

mkdir -p results &> /dev/null

parallel=0
for arg in "$@"
do
    if [ "${arg}" == "parallel" ]
    then
        parallel=1
    fi
done

for experiment in \
    gompertz_noise.sh \
    gompertz_sample.sh \
    gompertz_freq.sh \
    logistic_noise.sh \
    logistic_sample.sh \
    logistic_freq.sh \
    selkov_1.sh \
    selkov_2.sh \
    lorenz_1.sh \
    lorenz_2.sh \
    lorenz_3.sh \
    real_data.sh \
    fraction.sh \
    vi_sensitivity.sh \
    rebuttal.sh
do
    echo -n "Running ${experiment}"
    if [ ${parallel} -eq 1 ]
    then
        echo " in parallel"
        nohup bash ${experiment} &
    else
        echo " sequentially"
        bash ${experiment}
    fi
done
