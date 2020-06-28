import mxnet as mx

__all__ = []


def _register_lookahead_opt():
    optimizers = dict()
    for name in dir(mx.optimizer):
        obj = getattr(mx.optimizer, name)
        if hasattr(obj, '__base__') and obj.__base__ == mx.optimizer.Optimizer:
            optimizers[name] = obj
    prefix = 'Lookahead'

    def __init__(self, k=5, alpha=0.5, **kwargs):
        self.k = k
        self.alpha = alpha
        self._lookahead_params = dict()
        self._parent_cls = super(self.__class__, self)
        self._parent_cls.__init__(**kwargs)

    def update(self, index, weight, grad, state):
        self._lookahead_update_impl(
            index, weight, grad, state, self._parent_cls.update)

    def update_multi_precision(self, index, weight, grad, state):
        self._lookahead_update_impl(
            index, weight, grad, state, self._parent_cls.update_multi_precision)

    def _lookahead_update_impl(self, indexes, weights, grads, states, update_func):
        if not isinstance(indexes, (list, tuple)):
            indexes = [indexes]
            weights = [weights]
            grads = [grads]
        for index, weight in zip(indexes, weights):
            if index not in self._lookahead_params:
                self._lookahead_params[index] = weight.copy()
        update_func(indexes, weights, grads, states)
        for index, weight, grad in zip(indexes, weights, grads):
            count = self._index_update_count[index]
            if count % self.k == 0:
                old_weight = self._lookahead_params[index].as_in_context(weight.context)
                weight -= old_weight
                weight *= self.alpha
                old_weight += weight
                weight[:] = old_weight

    inst_dict = dict(
        __init__=__init__,
        update=update,
        update_multi_precision=update_multi_precision,
        _lookahead_update_impl=_lookahead_update_impl,
    )

    for k, v in optimizers.items():
        name = prefix + k
        inst = type(name, (v, ), inst_dict)
        mx.optimizer.Optimizer.register(inst)
        globals()[name] = inst
        __all__.append(name)


_register_lookahead_opt()
