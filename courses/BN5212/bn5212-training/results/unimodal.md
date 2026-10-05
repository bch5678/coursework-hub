| model | n | events | oof_auroc | auroc_lower | auroc_upper | fold_auroc_mean | fold_auroc_std | fold_auroc_min | fold_auroc_max | oof_auprc | auprc_lower | auprc_upper | oof_brier | test_n | test_events | test_auroc | test_auroc_lower | test_auroc_upper | test_auprc | test_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clinical-only (pretrained encoder) | 172 | 27 | 0.7254 | 0.6293 | 0.8154 | 0.7429 | 0.1085 | 0.6282 | 0.9226 | 0.3118 | 0.1926 | 0.4795 | 0.1250 | 26 | 4 | 0.6591 | 0.4571 | 0.8462 | 0.2396 | 0.1365 |
| Clinical-only (cohort only) | 172 | 27 | 0.6064 | 0.4596 | 0.7554 | 0.6687 | 0.1520 | 0.5032 | 0.8100 | 0.2601 | 0.1426 | 0.4062 | 0.1314 | 26 | 4 | 0.5909 | 0.2299 | 0.9231 | 0.2750 | 0.1391 |
| External clinical model (no cohort fitting) | 172 | 27 | 0.7476 | 0.6548 | 0.8328 | 0.7591 | 0.1078 | 0.6410 | 0.9355 | 0.3218 | 0.1983 | 0.4882 | 0.1296 | 26 | 4 | 0.6477 | 0.4526 | 0.8324 | 0.2313 | 0.1532 |
| CXR-only | 172 | 27 | 0.5517 | 0.4323 | 0.6795 | 0.5927 | 0.1367 | 0.4321 | 0.7700 | 0.1903 | 0.1154 | 0.3037 | 0.1377 | 26 | 4 | 0.6705 | 0.1238 | 1.0000 | 0.6042 | 0.1221 |
