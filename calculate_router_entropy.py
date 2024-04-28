from lm_polygraph.utils.manager import UEManager
from scipy.stats import entropy

import numpy as np
import torch
import os

# iterate over files in the polygraph directory
for file in os.listdir('polygraph'):
    man = UEManager.load('polygraph/' + file)
    for router_logits, greedy_tokens in zip(man.stats['router_logits'], man.stats['greedy_tokens']):
        num_generated_tokens = len(greedy_tokens)
        logits = np.stack(router_logits, axis=0).transpose(1,0,2)
        logits = logits[-num_generated_tokens:,:,:]
        
        entropies = torch.distributions.Categorical(torch.Tensor(logits).softmax(-1)).entropy().mean(0)
        
        for i in range(entropies.size(0)):
            man.estimations[('sequence', f'RouterEntropy{i}')].append(entropies[i].item())

    man.save('polygraph_with_layerwise_entropies/' + file)
