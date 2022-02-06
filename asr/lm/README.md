# Language Model 
- Dataset: Sherlock Holmes
- Hound: train/dev (80:20), test

## Logs
### Train_LM.ipynb
- huggingface + pytorch API
- epoch: 5
- train_loss': 6.13
- [!] Perplexity

### Train_LM_GRU_LSTM.ipynb
- Trax
- GRU, LSTM (respectively)

|      | log_ppl | ppl    |
|------|---------|--------|
| GRU  | 5.53    | 252.61 |
| LSTM | 5.56    | 285.63 |

### TODO
- Optimization
- Data augmentation?
- Transformer?
- Masking?
