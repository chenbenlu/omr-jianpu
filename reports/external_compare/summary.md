# External (photographed) evaluation: baseline vs fine-tuned

| model | split | n | SER | pitch_acc | rhythm_acc | edit_dist | gt_len |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | val | 100 | 0.5858 | 0.3353 | 0.8778 | 1325 | 2262 |
| baseline | test | 100 | 0.6343 | 0.3248 | 0.8617 | 1393 | 2196 |
| baseline | all | 200 | 0.6097 | 0.3302 | 0.8699 | 2718 | 4458 |
| finetuned | val | 100 | 0.2409 | 0.6686 | 0.9781 | 545 | 2262 |
| finetuned | test | 100 | 0.2582 | 0.6454 | 0.9648 | 567 | 2196 |
| finetuned | all | 200 | 0.2494 | 0.6574 | 0.9716 | 1112 | 4458 |
