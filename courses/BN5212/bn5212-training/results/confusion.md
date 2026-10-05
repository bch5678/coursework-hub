| model | evaluation | n | threshold_from | tn | fp | fn | tp | sensitivity | specificity | precision | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clinical-only | cross-validation | 172 | other folds | 89 | 56 | 10 | 17 | 0.6296 | 0.6138 | 0.2329 | 0.6163 |
| CXR-only | cross-validation | 172 | other folds | 102 | 43 | 18 | 9 | 0.3333 | 0.7034 | 0.1731 | 0.6453 |
| Clinical + CXR | cross-validation | 172 | other folds | 79 | 66 | 11 | 16 | 0.5926 | 0.5448 | 0.1951 | 0.5523 |
| Clinical-only | test | 26 | validation (out-of-fold) | 11 | 11 | 0 | 4 | 1.0000 | 0.5000 | 0.2667 | 0.5769 |
| CXR-only | test | 26 | validation (out-of-fold) | 7 | 15 | 1 | 3 | 0.7500 | 0.3182 | 0.1667 | 0.3846 |
| Clinical + CXR | test | 26 | validation (out-of-fold) | 13 | 9 | 0 | 4 | 1.0000 | 0.5909 | 0.3077 | 0.6538 |
