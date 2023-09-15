#!/usr/bin/env bash

declare -a splits=(testing validation training)

# Conflict point caching
# declare -a shards=(0 1 2 3 4 5 6 7 8 9)
# for j in "${shards[@]}"
# do
#    echo "Caching conflict points for ${1} split shard $j"
#    python conflict_points.py --split ${1} --shard_idx ${j}
# done
# echo "...done."

# Frenet+ features
# declare -a shards=(0 1 2 3 4 5 6 7 8 9)
# for j in "${shards[@]}"
# do
#    echo "*** Saving Frenetp features for ${1} split; running shard $j type: interp"
#    python compute_features.py --split ${1} --compute_frenetp_features --use_interp --shard_idx ${j} --cache
#    echo "*** Saving Frenetp features for ${1} split; running shard $j type: interp + hist"
#    python compute_features.py --split ${1} --compute_frenetp_features --use_interp --hist_only --shard_idx ${j} --cache
#    echo "*** Saving Frenetp features for ${1} split; running shard $j type: raw"
#    python compute_features.py --split ${1} --compute_frenetp_features --shard_idx ${j} --cache
# done
# echo "...done."


# Individual features
declare -a shards=(0 1 2 3 4 5 6 7 8 9)
for j in "${shards[@]}"
do
   echo "Caching individual features for ${1} split shard $j"
   python compute_features.py --split ${1} --compute_individual_features --shard_idx ${j} --cache --use_interp 
   python compute_features.py --split ${1} --compute_individual_features --shard_idx ${j} --cache --use_interp --use_extrap
done
echo "...done."