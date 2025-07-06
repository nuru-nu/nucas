Measurements from executing [`train_measure.ipynb`](./train_measure.ipynb)
on different platforms.

## Apple M2 Pro 16G

Running on pytorch 2.7.1

|              id |   steps |   sz |   batch_size | model_name   |   rollout_min |   rollout_max |      dt |
|----------------:|--------:|-----:|-------------:|:-------------|--------------:|--------------:|--------:|
| 20250706_221846 |    1000 |  128 |            4 | CaOrig       |            32 |            64 | 268.946 |
| 20250706_222701 |    1000 |  128 |            4 | CaOrig       |            32 |            64 | 260.768 |
| 20250706_223123 |    1000 |  128 |            4 | CaOrig       |            32 |            64 | 266.268 |


## Colab T4

|              id |   steps |   sz |   batch_size | model_name   |   rollout_min |   rollout_max |      dt |
|----------------:|--------:|-----:|-------------:|:-------------|--------------:|--------------:|--------:|
| 20250706_205720 |    1000 |  128 |            4 | CaOrig       |            32 |            64 | 221.771 |
| 20250706_210111 |    1000 |  128 |            4 | CaOrig       |            32 |            64 | 226.604 |
| 20250706_210504 |    1000 |  128 |            4 | CaOrig       |            32 |            64 | 227.903 |
