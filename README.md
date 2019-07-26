# Lookahead Optimizer: k steps forward, 1step back
It is a MXNet implementation of Lookahead Optimizer.

The link of the paper: https://arxiv.org/abs/1907.08610

## Usage
Import `optimizer.py`, then add the prefix `Lookahead` before the name of [arbitrary optimizer](http://mxnet.incubator.apache.org/api/python/optimization/optimization.html?highlight=opt#module-mxnet.optimizer).

```python
import optimizer
optimizer.LookaheadSGD(k=5, alpha=0.5, learning_rate=1e-3)
```

## Example
```bash
python mnist.py --optimizer sgd --seed 42
python mnist.py --optimizer lookaheadsgd --seed 42
```
