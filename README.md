# HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

For initial setup instructions look here - https://github.com/hotpotqa/hotpot

## Visualize Attention

Step 1: Record your attention activation values - Commit here - https://github.com/Jamil/HotpotQA/commit/396ceae8e20ac9ba7b959492754e3eed303d9006#diff-6e38f16215ae91c11fc5c54b74c66d54R236

Step 2: Convert those values to `numpy()` - `np.save('/tmp/test_att_old1.npy', outputs[0].cpu().detach().numpy())`

Step 3: Convert context_ids to `numpy()` - `np.save('/tmp/contextids_old1.npy', context_idxs.cpu().detach().numpy())`

Step 4: Visualize - https://github.com/Jamil/HotpotQA/blob/78b86843cb347856609875dd0ad83c5286421e74/visualize_attention.ipynb



## Results so far

biatt_before_start_token - Best F1 - 65.98

Gated CNN - Best F1  - 65.5

Self_att on both query and context - Best F1 - 62

Self_att on both query and context(shared weights) - Best F1 - ~61


## Learning rate and optimizations




