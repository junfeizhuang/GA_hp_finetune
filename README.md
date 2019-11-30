# Knowledge-distillation-example

This is an easy example to use genetic algorithm finding the deep network architecture for a batter classification accuracy.

It also fit for searching hyper parameters.

## Prepare

[gaft](https://github.com/PytLab/gaft)

```Script
pip install gaft
```

[mpi4py](https://pypi.org/project/mpi4py/1.1.0/)

```Script
pip install mpi4py
```

## Network search

We fix the convolution layers to find the reasonable channel number for two fully connected layers.

You can set the number of GPU to run with MPI.

## Run
```Script
mpiexec -n 1 python3 run.py -gpu 3
```
## Dataset

[Cifar10](http://www.cs.toronto.edu/~kriz/cifar.html)

## Log

The log name is best_fit.py show as below

```Python
best_fit = [
    (0, [113, 64], 13.168930053710938),
    (1, [100, 61], 17.49201202392578),
    (2, [103, 68], 15.625),
    (3, [105, 72], 17.53194808959961),
    (4, [110, 63], 21.026357650756836),
]
```