#!/bin/bash

TOPPS=("0.3" "0.4" "0.5" "0.6" "0.65" "0.7" "0.75" "0.8" "0.85" "0.9" "0.95" "1.0")

for shot in $(seq 4 5); do
  SAMPLE_EXPERTS=0 HYDRA_CONFIG=`pwd`"/examples/configs/polygraph_eval_medmcqa_new_prompt_${shot}.yaml" polygraph_eval task="mixtral_base_${shot}_shot";
  
  for topp in "${TOPPS[@]}"; do
  	SAMPLE_EXPERTS=1 TOPX=topp TOPP=$topp HYDRA_CONFIG=`pwd`"/examples/configs/polygraph_eval_medmcqa_new_prompt_${shot}.yaml" polygraph_eval task="mixtral_topp_${topp}_${shot}_shot";
  done
  
  for topk in $(seq 1 8); do
  	SAMPLE_EXPERTS=1 TOPX=topk TOPK=$topk HYDRA_CONFIG=`pwd`"/examples/configs/polygraph_eval_medmcqa_new_prompt_${shot}.yaml" polygraph_eval task="mixtral_topk_${topk}_${shot}_shot";
  done
done
