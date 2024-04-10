import io
import json
import copy
import numbers
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud

import postgresql_db as db
import extra_functions as ef
import fit_utils as fu


from statistics import mean as st_mean
from datetime import datetime
from itertools import product, compress
from special_dicts import NestedDict, NestedDictList
from m_classes import SELF, FunctionWrapper


def get_default_device(device = 'cuda'):
    """Pick GPU if available, else CPU"""
    if (device.lower() == 'cuda' or device.lower() == 'gpu') and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Function that can move data and model to a chosen device.
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


dump_bytes_io = fu.save_to_bytes_io(torch.save)



class DeviceDataLoader(tud.DataLoader):
    """Wrap a dataLoader to move data to a device"""
    def __init__(self, device, *args, **kwargs):
        self.device = device
        super(DeviceDataLoader, self).__init__(*args, **kwargs)
    def __iter__(self):
        x = super(DeviceDataLoader, self).__iter__()
        for y in x:
            yield to_device(y, self.device)
        return x

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class TorchMetric(fu.Metric):
    # @staticmethod
    # condition_conversion = lambda x, pred: x.type(pred.dtype)

    def append(self, pred, y):
        self.pred = (torch.cat([pred])
                     if self.pred is None
                     else torch.cat([self.pred, pred])).detach()
        self.y = (torch.cat([y])
                  if self.y is None
                  else torch.cat([self.y, y])).detach()
    def clear(self):
        self.pred = None
        self.y = None

    def compute(self, mean=None, clear=True):
        out = self(self.pred, self.y, mean=mean)
        if clear:
            self.clear()
        if out.size():
            return out.detach().cpu().numpy()
        else:
            return out.item()
    @staticmethod
    def mean(inp, mean=None):
        return inp.mean(mean)
    # @staticmethod
    def top_k(self, inp, frac=SELF('top_k_frac')):
        frac = SELF.check_get(frac, self)
        return inp.topk(int(len(inp) * frac), dim=0)

    def __call__(self, pred, y, mean=None):
        raise NotImplementedError


class MeanAbsoluteError(fu.MeanAbsoluteError, TorchMetric):
    pass

class SignHitPercent(fu.SignHitPercent, TorchMetric):
    pass

class MeanOutputRatio(fu.MeanOutputRatio, TorchMetric):
    pass

class MeanAbsoluteSignal(fu.MeanAbsoluteSignal, TorchMetric):
    pass


def param_tuples(model):
    return [(name[0] + \
            name.split('.')[0].replace(name.split('.')[0].rstrip('0123456789'),'') + \
            '.' + ','.join(map(str,v)), p.data[v].item()) \
            for name,p in list(model.named_parameters()) \
            for v in product(*[range(i) for i in p.shape])]

def evaluate(model, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)



def fit(model, epochs, train_loader, test_loader_epoch, test_loader_final,
        device=None, loss=None, optimizer=None, scheduler=None, metric=None, loader_fn_kw=None,
        keepers=(), **kwargs):

    def train_lambda(loader, _model=model, **kw):
        return train(_model, loader, device=device, loss_fn=loss, optimizer=optimizer, scheduler=scheduler, prints=(kwargs | kw).get('epoch_print', False))

    # def test_lambda(loader, _model=model, **kw):
    #     return test(_model, loader, device=device, loss=loss, metric=metric, **kw)

    test_lambda = ef.FunctionArgBinder(test, device=device, loss=loss, metric=metric, kwargs2args=dict(_model=model), arg_bound_first=True)
    # test_lambda = lambda loader, _model=model, **kw: test(_model, loader, device=device, loss=loss, metric=metric, **kw)

    return fu.fit_shell(train_lambda, test_lambda, epochs, train_loader, test_loader_epoch,
                        test_loader_final, loader_fn_kw=loader_fn_kw, keepers=keepers, _model=model, **kwargs)

def train(model, data_loader, device=get_default_device(), loss_fn=None, optimizer=None, scheduler=None, prints=False):
    if loss_fn is None:
        loss_fn = model.loss
    if optimizer is None:
        optimizer = model.optimizer


    # size = len(data_loader.dataset)
    # if model.is_frozen_ensemble:
    #     data_loader = model.prepare_meta_dataloader(data_loader)
    #     model = model.meta_model
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)
        # print(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_step = False
        if hasattr(scheduler, 'batch_step') and scheduler.batch_step:
            scheduler.step()
            batch_step = True
    lr = get_lr(optimizer)
    model.config['lr'] = lr
    if prints:
        print(lr)
    if scheduler is not None and not batch_step:
        scheduler.step()
    return model


def test(model, data_loader, device=get_default_device(), loss=None, metric=None, **kwargs):
    if model is not None:
        model.eval()
    # if model.is_frozen_ensemble:
    #     data_loader = model.prepare_meta_dataloader(data_loader)
    #     model = model.meta_model
    if loss is None:
        loss = model.loss
    if metric is None and hasattr(model, 'metric'):
        metric = [model.metric]
    def data_loader_loop(_metric):
        loss_v = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss_v += loss(pred, y).item()
                if _metric is not None:
                    for m in _metric:
                        m.append(pred, y)
        loss_v /= len(data_loader)
        return loss_v

    return fu.test_shell(data_loader_loop, metric, **kwargs)


def ts_make_ds(*args, dtype=None, **kwargs):
    if dtype is None:
        dtype = torch.float32
    def func(x, y):

        X_train = torch.from_numpy(x).type(dtype)
        y_train = torch.from_numpy(y).type(dtype)
        return tud.TensorDataset(X_train, y_train)
    return fu.ts_make_ds(*args, func=func, **kwargs)


def nested_helper_func(level=None, batch_size=64, debug=False, agg_level=1, accessor=lambda x: x, random_state=None):
    f_c = FunctionWrapper.create_constructor
    test_f_fn = f_c(lambda x, f: tud.random_split(accessor(x), [f, 1 - f],
                                                  generator=torch.Generator().manual_seed(random_state) if random_state is not None else None),
                    update=False, level=level)
    lmbd_dl = lambda x: tud.DataLoader(x, batch_size, shuffle=False)
    dl_fn = f_c(lambda x: tuple(lmbd_dl(y) for y in x) if isinstance(x, list) else lmbd_dl(x), update=False)
    agg_fn = f_c(tud.ConcatDataset, agg_level=agg_level, update=False)
    # dl_fn = lambda y: FunctionWrapper(lambda x, **kw: x.fn(tud.DataLoader, batch_size,
    #                                                        shuffle=False, update=False), target = y)
    # agg_fn = lambda x, **kw: x.fn(tud.ConcatDataset, agg_level=agg_level, update=False, **kw if not debug else {})
    return test_f_fn, dl_fn, agg_fn



# def fit2(model, epochs, train_loader, test_loader, device, loss_fn=None,
#          optimizer=None, metric_fn=None, dbPackage=None):
#     """Train the model using gradient descent"""
#     def dumps(j): return json.dumps(j, sort_keys=True, default=str)
#     history = None
#     time1 = datetime.now()
#     _test = lambda loader: test(model, loader, device=device, loss=loss_fn, metric=metric_fn)
#
#     if isinstance(test_loader, NestedDict):
#         result = _test
#     else:
#         result = test(model, test_loader, device=device, loss=loss_fn, metric=metric_fn)
#     time2 = datetime.now()
#     dbOut = {str(k) + '_0':v for k,v in result.items()}
#
#     if dbPackage is not None:
#         vals = dbPackage['values']
#         def upload(model, optimizer, epoch, t2, t1):
#             zip_param = list(zip(*param_tuples(model)))
#             col = list(vals.keys()) + ["binary", "binaryKeys", "json_hr", "json"] + list(zip_param[0])
#             cmp_insert = db.composed_insert(dbPackage['table'], col, schema = 'dt',
#                                             returning = ['uuid'])
#             bn = {'model_state_dict': model.state_dict(), 'optimizer': optimizer,
#                     'optimizer_state_dict': optimizer.state_dict()}
#             jshr = result | {'epoch': 0, 'epochs': epochs,
#                             'epoch_time': (t2 - t1).total_seconds()}
#             js = {'optimizer': optimizer} | jshr
#             val = [tuple(vals.values()) + (save_io(bn).read(), list(bn.keys()).sort(),
#                                         dumps(jshr), dumps(js)) + zip_param[1]]
#             db.pg_execute(dbPackage['conn'], cmp_insert, val)
#         upload(model, optimizer, 0, time2, time1)
#
#     for epoch in range(1, epochs+1):
#         time3 = datetime.now()
#         train(model, train_loader, device=device, loss_fn=loss_fn, optimizer=optimizer)
#         result = test(model, test_loader, device=device, loss=loss_fn, metric=metric_fn)
#         time4 = datetime.now()
#
#         if dbPackage is not None:
#             upload(model, optimizer, epoch, time2, time1)
#
#         # model.epoch_end(epoch, result)
#         if history is None:
#             history = result.copy()
#             history.update([(x, [y]) for x, y in result.items()])
#         else:
#             history.update([(x, history[x] + [y]) for x, y in result.items()])
#         dbOut = dbOut | result
#
#     return history, dbOut
