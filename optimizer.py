import mxnet as mx


@mx.optimizer.Optimizer.register
class LookaheadSGD(mx.optimizer.SGD):
    def __init__(self, k=5, alpha=0.5, **kwargs):
        self.k = k
        self.alpha = alpha
        self.lookahead_params = dict()
        super(LookaheadSGD, self).__init__(**kwargs)
        self._update = super(LookaheadSGD, self).update
        self._update_multi_precision = super(LookaheadSGD, self).update_multi_precision
    def update(self, index, weight, grad, state):
        self._lookahead_update_impl(index, weight, grad, state, self._update)
    def update_multi_precision(self, index, weight, grad, state):
        self._lookahead_update_impl(index, weight, grad, state, self._update_multi_precision)
    def _lookahead_update_impl(self, indexes, weights, grads, states, update_func):
        for index, weight in zip(indexes, weights):
            if index not in self.lookahead_params:
                self.lookahead_params[index] = weight.copy()
        update_func(indexes, weights, grads, states)
        for index, weight, grad in zip(indexes, weights, grads):
            count = self._index_update_count[index]
            if count % self.k == 0:
                old_weight = self.lookahead_params[index]
                weight -= old_weight
                weight *= self.alpha
                old_weight += weight
                weight[:] = old_weight
