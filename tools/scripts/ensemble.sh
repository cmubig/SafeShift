#!/usr/bin/env bash

# Examples: 
#    bash scripts/ensemble.sh bootstrap mtr+20p_initial_bootstrap 48
#    bash scripts/ensemble.sh new_waymo mtr+20p_initial 48

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    exit 2
fi

BASE=$1
META=$2
BATCH=$3
cfgs=$(python -c "import os; print(' '.join(sorted([x for x in os.listdir('cfgs/${BASE}') if x.endswith('_test.yaml')])))")
for var in ${cfgs};
    do bash scripts/dist_test.sh 4 --cfg_file cfgs/${BASE}/${var} --ckpt \
            ../output/${BASE}/${var%_test.yaml}/default/ckpt/best_model.pth --save_train --save_val --batch_size ${BATCH}
done

python label_classifier.py --cfg_file cfgs/meta/${META}.yaml
python ensemble_res.py --cfg_file cfgs/meta/${META}.yaml