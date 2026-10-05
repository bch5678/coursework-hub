| model | n | events | oof_auroc | auroc_lower | auroc_upper | fold_auroc_mean | fold_auroc_std | fold_auroc_min | fold_auroc_max | oof_auprc | auprc_lower | auprc_upper | oof_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ImageNet ViT-B/16, frozen (kept) | 172 | 27 | 0.5517 | 0.4323 | 0.6795 | 0.5927 | 0.1367 | 0.4321 | 0.7700 | 0.1903 | 0.1154 | 0.3037 | 0.1377 |
| ViT-Tiny, last 2 blocks fine-tuned | 172 | 27 | 0.4953 | 0.3702 | 0.6149 | 0.4866 | 0.1528 | 0.2593 | 0.6387 | 0.1651 | 0.0996 | 0.2563 | 0.1361 |
| DenseNet121 CheXpert, frozen | 172 | 27 | 0.4784 | 0.3519 | 0.6033 | 0.5168 | 0.1813 | 0.2654 | 0.7000 | 0.1478 | 0.0888 | 0.2304 | 0.1381 |
| ViT-Tiny, frozen | 172 | 27 | 0.4623 | 0.3207 | 0.5966 | 0.4780 | 0.1484 | 0.2593 | 0.6387 | 0.1610 | 0.0970 | 0.2499 | 0.1371 |
