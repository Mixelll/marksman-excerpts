import fit_utils as fu
import numpy as np
import operator as op
import joblib as jl
import extra_functions as me

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_poisson_deviance

from m_classes import SELF, FunctionWrapper

dump_bytes_io = fu.save_to_bytes_io(jl.dump)

# def dump(model, path):
#     jl.dump(model, path)
#
# def load(path):
#     return jl.load(path)

def criterion_to_metric(criterion):
    mapping = {
    'squared_error': mean_squared_error,
    'absolute_error': mean_absolute_error,
    # 'friedman_mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred) / mean_squared_error(y_true, np.mean(y_true)),  # Corrected mapping for 'friedman_mse'
    'poisson': mean_poisson_deviance,
    # Add more mappings as needed
    }
    return mapping.get(criterion)

class SKMetric(fu.Metric):
    def append(self, pred, y):
        self.pred = np.concatenate([self.pred, pred]) if self.pred is not None else pred
        self.y = np.concatenate([self.y, y]) if self.y is not None else y

    def clear(self):
        self.pred = None
        self.y = None

    def compute(self, mean=None, clear=True):
        out = self(self.pred, self.y, mean=mean)
        out = out.tolist() if isinstance(out, np.ndarray) else out
        # super().add(out)
        if clear:
            self.clear()
        return out

    @staticmethod
    def mean(inp, mean=None):
        return inp.mean(axis=mean)

    def top_k(self, inp, frac=SELF('top_k_frac')):
        frac = SELF.check_get(frac, self)
        k = int(len(inp) * frac)
        indices = np.argpartition(inp, -k, axis=0)[-k:]
        return inp[indices], indices

    def __call__(self, pred, y, mean=None):
        raise NotImplementedError


class MeanAbsoluteError(fu.MeanAbsoluteError, SKMetric):
    pass

class SignHitPercent(fu.SignHitPercent, SKMetric):
    pass

class MeanOutputRatio(fu.MeanOutputRatio, SKMetric):
    pass

class MeanAbsoluteSignal(fu.MeanAbsoluteSignal, SKMetric):
    pass

def fit(model, epochs, train_loader, test_loader_epoch, test_loader_final, metric=None, device=None, optimizer=None, **kwargs):
    if device is not None and device != 'cpu':
        raise NotImplementedError
    if optimizer is not None:
        raise NotImplementedError
    train_lambda = lambda loader: train(model, loader)
    test_lambda = lambda loader, **kw: test(model, loader, metric=metric, **kw)
    return fu.fit_shell(train_lambda, test_lambda, epochs, train_loader,
                        test_loader_epoch, test_loader_final, zero_result_kw={'fitted': False} **kwargs)

def train(model, train_loader):
    model.fit(train_loader[0][:,:,0], train_loader[1].ravel())
    return model


def test(model, data_loader, loss=None, metric=None, fitted=True, **kwargs):
    if loss is None:
        loss_fetched = criterion_to_metric(model.criterion if 'criterion' in dir(model) else None)
        loss = loss_fetched if loss_fetched is not None else criterion_to_metric('squared_error')
    if metric is None and hasattr(model, 'metric'):
        metric = [model.metric]
    def data_loader_loop(_metric):
        x, y = data_loader
        x, y = x[:,:,0], y.ravel()
        pred = model.predict(x)
        loss_v = loss(pred, y) if loss is not None else None
        if _metric is not None:
            # pred = model['scaler'].inverse_transform(pred)
            for m in _metric:
                m.append(pred, y)
        return loss_v

    return fu.test_shell(data_loader_loop, metric, return_structure=not fitted, **kwargs)

@me.copy_signature(fu.ts_make_ds)
def ts_make_ds(*args, **kwargs):
    # signature(forwards).bind(*args, **kwargs)
    return fu.ts_make_ds(*args, **kwargs)

def nested_helper_func(level=4, batch_size=64, debug=False, agg_level=1, accessor=lambda x: x, random_state=None):
    f_c = FunctionWrapper.create_constructor
    l = lambda x, f: op.itemgetter(*[1,3])(train_test_split(*accessor(x), test_size=f, random_state=random_state))
    test_f_fn = f_c(l, update=False, level=level, ignore_none=False)
    dl_fn = f_c(lambda x: x, update=False, supress_kw=debug)
    ll = lambda ls: tuple(np.concatenate([x[i] for x in ls]) for i in range(len(ls[0])))
    agg_fn = f_c(ll, agg_level=agg_level, update=False, level=level, supress_kw=debug)
    return test_f_fn, dl_fn, agg_fn

def split(ds, test_size, random_state=None):
    return op.itemgetter(*[1, 3])(train_test_split(*ds, test_size=test_size, random_state=random_state))