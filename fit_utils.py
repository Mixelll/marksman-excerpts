import functools
import io
import json
import numbers
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from itertools import compress, product
import statistics as st

import numpy as np
import pandas as pd

import extra_functions as ef
import postgresql_db as db
from m_classes import FunctionWrapper, SELF
from special_dicts import NestedDict, NestedDictList


def to_str(model):
    if hasattr(model, 'str'):
        return model.str()
    else:
        return str(model)


def save_to_bytes_io(func):
    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        buffer = io.BytesIO()
        func(obj, buffer, *args, **kwargs)
        buffer.seek(0)
        return buffer
    return wrapper


class Instancer:
    def __init__(self, obj, *args, permute_kwargs=None, **kwargs):
        self.obj = obj
        self.args = args
        self.kwargs = kwargs
        self.permute_kwargs = permute_kwargs

    def __call__(self):
        if self.permute_kwargs is not None:
            return [self.obj(*self.args, **self.kwargs, **kw) for kw in self.permute_kwargs]
        return self.obj(*self.args, **self.kwargs)


class MetricInstancer(Instancer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        called = self()
        if isinstance(called, (list, tuple)):
            self.name = [c.name for c in called]
            self.name2 = [c.name2 for c in called]
        else:
            self.name = called.name
            self.name2 = called.name2


class Metric:
    name = None
    name2 = None
    alias = None
    pred = None
    y = None
    last = None
    len = None
    meta_metric = False
    # defaults = (('meta_metric', False),)

    def __init__(self, top_k_frac=None, top_k_func=abs, top_k_source_y=True):
        self.top_k_frac = top_k_frac
        self.top_k_func = top_k_func
        self.top_k_source_y = top_k_source_y
        self.name_add = '' if top_k_frac is None else f'_{round(100 * self.top_k_frac)}'
        self.name = self.name + self.name_add
        self.name2 = self.name2 + self.name_add

        # for k, v in self.defaults:
        #     if not hasattr(self, k):
        #         setattr(self, k, v)

        if self.meta_metric:
            self.source_names = [x + self.name_add for x in self.source_names]

    def _post_call(self, out):
        self.last = out

    def transform_pred_y(self, pred, y, frac=SELF('top_k_frac'), top_k_source_y=SELF('top_k_source_y')):
        frac = SELF.check_get(frac, self)
        top_k_source_y = SELF.check_get(top_k_source_y, self)
        if frac is not None:
            source = y if top_k_source_y else pred
            source = self.top_k_func(source) if self.top_k_func is not None else source
            _, ind = self.top_k(source, frac=frac)
            ind = ind.squeeze()
            pred, y = pred[ind], y[ind]
        return pred, y


def agg_nested(agg_dict, result_level=1, key=None, filter_none_list=False, update_mean=False, discard_none=True,
               train_ds_nesting=None,  **kwargs):
    ax = 1
    if train_ds_nesting is not None:
        agg_dict = agg_dict.copy().del_n(train_ds_nesting)
    agg = agg_dict.change_nesting(pop_insert=[-2, -2], inner_out=1)
    lvl = -result_level - 1
    agg_lvl = agg.count_levels() - 2
    list_fn = lambda x: np.concatenate(x, axis=ax) if isinstance(x, np.ndarray) else list(x)
    agg.fn(list_fn, agg_level=lvl if abs(lvl) < agg_lvl else -agg_lvl)
    if filter_none_list:
        cond_y = lambda y: not any(x is None for x in y)
    else:
        cond_y = lambda y: True
    f = lambda x, min_len=1, **kw: agg.fn(lambda y: x(y, **kw) if isinstance(y, Iterable) and len(y) >= min_len and cond_y(y) else None,
                                          discard_none=discard_none, update=False)
    mean_fn = lambda x, **_kw: np.mean(x, **_kw) if isinstance(x, np.ndarray) else st.mean(x)
    std_fn = lambda x, **_kw: np.std(x, **_kw) if isinstance(x, np.ndarray) else st.stdev(x)
    min_fn = lambda x, **_kw: np.min(x, **_kw) if isinstance(x, np.ndarray) else min(x)
    max_fn = lambda x, **_kw: np.max(x, **_kw) if isinstance(x, np.ndarray) else max(x)
    out = NestedDict({'mean': f(mean_fn), 'stdev': f(std_fn, min_len=2), 'min': f(min_fn), 'max': f(max_fn)})

    # out = NestedDict({'mean': f(np.mean, axis=ax), 'stdev': f(np.std, axis=ax, min_len=2), 'min': f(np.min, axis=ax), 'max': f(np.max, axis=ax)})

    if key is not None:
        out = out[key]
    if update_mean:
        out['mean'].update_new(agg)
    return out.change_nesting(outer_in=1)


def agg4keeper_merged(result, delete_keys=('val',), **kwargs):
    result = result.copy().delete_keys(delete_keys)
    return agg_nested(result, **kwargs).fn(lambda x: x['mean'], level=-1, update=False).change_nesting(outer_in=1)


class KeeperObj:
    def __init__(self, *args, merged_keeper_modes=(4,), test_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_keeper_modes = merged_keeper_modes
        self.test_fn = test_fn

    # def create(self):
    #     new_obj = self.copy()
    #     new_obj.clear()
    #     for i, kb in self.iter():
    #         new_obj[i] = kb() if not isinstance(kb, BestKeeper) else kb
    #     return new_obj

    def create(self, name=None):
        new_obj = self.copy()
        if name is not None:
            new_obj.name = name
        return new_obj.fn(lambda kb: kb() if not isinstance(kb, BestKeeper) else kb)

    def build_nested(self, zero_result, merged_keeper_modes=SELF('merged_keeper_modes'), merged_kw=None):
        merged_keeper_modes = SELF.check_get(merged_keeper_modes, self)
        if merged_keeper_modes is None:
            merged_kw = {}
        levels = zero_result.count_levels()
        def build_kb(kb):
            kb0 = deepcopy(kb)
            kb.agg_dict = zero_result.fn(lambda: deepcopy(kb0), level=-kb.result_nesting_level, update=False)
            if merged_keeper_modes:
                kw = dict(level=-kb.result_nesting_level, update=False)
                mode_fn = lambda x: agg4keeper_merged(zero_result, result_level=x, filter_none_list=True, discard_none=False).fn(
                    lambda: deepcopy(kb0), **kw)
                kb.merged_keeper = NestedDict({f'level_{mode}': mode_fn(mode) for mode in merged_keeper_modes})
                kb.merged_keeper_modes = merged_keeper_modes

        if levels > 1:
            self.fn(build_kb, update=False)
        return

    def check_keep(self, model, results_dict, copy=False, **kwargs):
        if self.fn(lambda kb: kb.check(results_dict), agg_level=0, agg_fn=any, update=False):
            if copy:
                if hasattr(model, 'copy'):
                    model_keep = model.copy()
                else:
                    model_keep = deepcopy(model)
            else:
                model_keep = model
            named_list = [self.name] if self.name is not None else []
            model_keep.keep_history.set_n(named_list + ['result'], results_dict)
            self.fn(lambda kb, nesting, /: kb.check_keep(model_keep, results_dict, copy=False, nesting=named_list+['kept_by']+list(nesting), **kwargs),
                    update=False)
        # if any([kb.check(results_dict) for kb in self]):
        #     if copy:
        #         if hasattr(model, 'copy'):
        #             model_keep = model.copy()
        #         else:
        #             model_keep = deepcopy(model)
        #     else:
        #         model_keep = model
        #     for kb in self:
        #         # print(f'AAA:{kb.agg_dict.ival(0).agg_dict}')
        #         kb.check_keep(model_keep, results_dict, copy=False, **kwargs)

    def test_check_keep(self, models, return_results=False, **kwargs):
        if not isinstance(models, (list, tuple)):
            models = [models]

        if return_results:
            results_out = NestedDictList()

        for model in models:
            results_dict = self.test_fn(model)
            if return_results:
                results_out.append(results_dict)
            self.check_keep(model, results_dict, **kwargs)

        if return_results:
            return results_out
        return self

    def get_models(self, return_list=None, get_results=False):
        if return_list is None:
            return_list = []

        self.fn(lambda kb: kb.get_models(return_list, get_results=get_results), update=False)
        # for kb in self:
        #     kb.get_models(return_list)
        return ef.unique(return_list)

    def compare_bests(self, other, **kwargs):
        return self.func_with_other(lambda x, y: x.compare_bests(y, **kwargs), other)

    def _clear(self):
        self.fn(lambda kb: kb.clear(), update=False)

    def clear_get_models(self, return_list=None):
        models = self.get_models(return_list)
        self._clear()
        return models


class KeeperDict(KeeperObj, NestedDict):
    def get_best(self):
        return self.fn(lambda kb: kb.get_best(), update=False)

    def best_agg(self):
        return self.fn(lambda kb: kb.best_agg(), update=False)


class KeeperList(KeeperObj, NestedDictList):

    def get_best(self):
        return NestedDict({kb.name: kb.get_best() for kb in self})

    def best_agg(self):
        return NestedDict({kb.name: kb.best_agg() for kb in self})


class BestKeeper:
    agg_dict = None
    merged_keeper = None
    # agg_level = None

    def __init__(self, metric=None, best=None, model=None, list_fn=all, result_nesting_level=1, name=None,
                 result_bool=None, test_fn=None, n_best=1):
        self.metric = metric
        self.best = best
        self.model = model
        self.list_fn = list_fn
        self.result_nesting_level = result_nesting_level
        self.name = name
        if result_bool is None and isinstance(self.metric, (list, tuple)):
            self.result_bool = any(m.result_bool for m in self.metric)
        else:
            self.result_bool = result_bool

        # if self.metric is None:
        #     self.metric = self

        if test_fn is not None:
            self.test_fn = test_fn

        self.models_result_deque = ef.SortedDeque(maxlen=n_best, compare=self.metric_output_fn, accessor=ef.AccessorK(1),
                                                  assert_fn=self.bool_metric_fn if self.result_bool else None)

    def metric_output_fn(self, _result, _best):
        if isinstance(self.metric, (list, tuple)):
            return self.list_fn([m(_result, _best) for m in self.metric])
        else:
            return self.metric(_result, _best)

    def bool_metric_fn(self, result):
        mix_bool_out = self.list_fn([m(result) for m in self.metric if m.result_bool])
        if isinstance(mix_bool_out, (list, tuple)):
            for x in mix_bool_out:
                if isinstance(x, bool):
                    if not x:
                        return False
            return True
        else:
            return mix_bool_out

    def get_best(self):
        if self.agg_dict is not None:
            return self.agg_dict.fn(lambda x: x.best, update=False)
        else:
            return self.best

    def compare_bests(self, other, compare_fn=lambda x, y: x - y, iterate=True):
        if isinstance(other, BestKeeper):
            other = other.get_best()
        if iterate:
            c_f = compare_fn
            _compare_fn = lambda x, y: [c_f(xi, yi) for xi, yi in zip(x, y)] if all(map(lambda z: isinstance(z, Iterable), [x, y])) else c_f(x, y)
        return self.get_best().fn_other(other, _compare_fn, update=False).set_name(self.name)

    def best_agg(self):
        if self.agg_dict is not None:
            res_lvl = self.result_nesting_level
            return agg_nested(self.agg_dict.fn(lambda x: x.best, update=False), result_level=res_lvl, update_mean=True)
        else:
            return self.best

    def check(self, result, return_nested=False, model=None):

        if self.agg_dict is not None:
            # aggregate_level
            out = self.agg_dict.fn(lambda x, nesting, /: x.check(result.get_n(nesting)), update=False)
            if return_nested:
                return out
            else:
                merged_res = NestedDict({4: agg4keeper_merged(result, result_level=4, train_ds_nesting=model.config['train_ds'] if model is not None else None)})
                if self.merged_keeper is not None:
                    return out.any() | self.merged_keeper.fn(lambda x, nesting, /: x.check(merged_res.get_n(nesting)), update=False).any()
                return out.any()

        return self.models_result_deque.insert_sorted((None, result), check_only=True,)

    def keep(self, model, result, copy=False, add_tuples=()):
        if copy:
            _model = model.copy()
        else:
            _model = model
        self.models_result_deque.insert_sorted(tuple([_model, result] + list(add_tuples)), check_only=False)

    def check_keep(self, model, result, copy=False, nesting=None, **kwargs):
        nesting_ls = list(nesting) if nesting is not None else []
        if self.agg_dict is not None or self.merged_keeper is not None:
            if self.merged_keeper is not None:
                merged_res = NestedDict({f'level_{mode}': agg4keeper_merged(result, result_level=mode, train_ds_nesting=model.config['train_ds'])
                                         for mode in self.merged_keeper_modes})
                # merged_res = NestedDict({4: agg4keeper_merged(result, result_level=4)})

                self.merged_keeper.fn(lambda x, _nesting, /: x.check_keep(model, merged_res.get_n(_nesting), copy=copy, nesting=nesting_ls+list(_nesting), **kwargs), update=False)
            if self.agg_dict is not None:
                return self.agg_dict.fn(lambda x, _nesting, /: x.check_keep(model, result.get_n(_nesting), copy=copy, nesting=nesting_ls+list(_nesting), **kwargs), update=False)
        else:
            if self.check(result, model=model):
                model.keep_history.set_n(nesting_ls + [self.name], result)
                self.keep(model, result, copy=copy, **kwargs)
                return True
            else:
                return False

    def get_models(self, return_list=None, get_results=False):
        def nested_get_models(y):
            return y.fn(lambda x, nesting, /: x.get_models(return_list), update=False)

        if return_list is None:
            return_list = []
        if len(self.models_result_deque):
            if get_results:
                return_list.extend(tuple(self.models_result_deque))
            else:
                return_list.extend(ef.get_sub_lists_i(self.models_result_deque, i=0))
        if self.agg_dict is not None:
            nested_get_models(self.agg_dict)
            nested_get_models(self.merged_keeper)
        if get_results:
            return ef.unique(return_list, accessor=ef.AccessorK(0), aggregate=True, aggregate_mode=0)
        return ef.unique(return_list)

    def clear(self):
        if self.agg_dict is not None:
            self.agg_dict.fn(lambda x, nesting, /: x.clear(), update=False)
            if self.merged_keeper is not None:
                self.merged_keeper.fn(lambda x, nesting, /: x.clear(), update=False)
        self.models_result_deque.clear()

    def clear_get_models(self, return_list=None):
        if return_list is None:
            return_list = []
        self.get_models(return_list)
        self.clear()
        return return_list

    def __call__(self, model, result, **kwargs):
        self.check_keep(model, result, copy=True, **kwargs)


class MetricComparer:
    def __init__(self, metric_or_target_function, key=None, func=None, result_bool=False,
                 greater_is_better=True, compare_fn=None, add_name=None, name2=None):
        self.metric = metric_or_target_function
        self.key = key.name if isinstance(key, (Metric, MetricInstancer)) else key
        self.func = func
        self.result_bool = result_bool
        self.greater_is_better = greater_is_better
        self.compare_fn = compare_fn
        self.name = '' if key is None else self.key
        self.name2 = name2 if name2 is not None else key.name2 if isinstance(key, (Metric, MetricInstancer)) else self.name

        _add_name = '' if add_name is None else f'_{add_name if isinstance(add_name, str) else "_".join(add_name)}'
        self.name += _add_name
        self.name2 += _add_name

    def __call__(self, result, best=None):
        metric = lambda x: self.metric((x if self.func is None else self.func(x)) if self.key is None else (x[self.key] if self.func is None else self.func(x[self.key])))
        # def metric(x):
        #     return self.metric(self.func(x) if self.key is None else self.func(x[self.key]))

        if self.result_bool:
            return metric(result)

        assert best is not None
        result_metric = metric(result)
        best_metric = metric(best)
        if self.compare_fn is  None:
            if self.greater_is_better:
                return result_metric > best_metric
            else:
                return result_metric < best_metric
        else:
            return self.compare_fn(result_metric, best_metric)


class TargetFunction:
    def __init__(self, col_str, func=st.mean, keys=None, keys_arr=None, separator='_', map_func=None, return_vec=False):
        if return_vec:
            func = lambda x: x
        self.col_str = col_str
        self.func = func
        self.map_func = map_func
        self.keys = keys
        self.keys_arr = keys_arr
        self.separator = separator

    def _map_func(self, x):
        if self.map_func is not None:
            return map(self.map_func, x)
        return x
    def __call__(self, col_tr, keys=SELF('keys'), keys_arr=SELF('keys_arr'),
                    separator=SELF('separator'), _bool_flag=False, **kwargs):
        keys = SELF.check_get(keys, self)
        keys_arr = SELF.check_get(keys_arr, self)
        separator = SELF.check_get(separator, self)

        if isinstance(col_tr, numbers.Number):
            return self.func(self._map_func([col_tr]))
        elif keys is None and keys_arr is None:
            return self.func(self._map_func(col_tr))
        elif keys_arr is not None:
            loc_bools = [False] * len(col_tr)
            for _keys in keys_arr:
                loc_bools |= self(self.col_str, col_tr, keys=_keys, keys_arr=None,
                                             separator=separator, _bool_flag=True, **kwargs)
            return self.func(self._map_func(compress(col_tr, loc_bools)))
        elif keys is not None:
            loc_obj = pd.MultiIndex.from_tuples(map(lambda x: x.split(separator), self.col_str)).get_loc(tuple(keys))
            if isinstance(loc_obj, int):
                if _bool_flag:
                    out = [False] * len(col_tr)
                    out[loc_obj] = True
                    return out
                else:
                    return col_tr[loc_obj]
            else:
                return loc_obj


def get_result_dict(loaders, test_lambda, test_names=None, func=None, **kwargs):
    if func is None:
        func = lambda x: x
    if test_names is None:
        test_names = {}
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]

    def get_result(loader_optional_nested, i=None, **kw):

        if isinstance(loader_optional_nested, dict):
            loader_optional_nested = NestedDict.from_dict(loader_optional_nested)
        if isinstance(loader_optional_nested, NestedDict):
            return loader_optional_nested.fn(
                lambda x, nes, /, **_kw: test_lambda(func(x), name=nes[-1 if len(nes) == 1 else -2], **_kw),
                update=False, **kw)
        else:
            if i is not None:
                return {i: test_lambda(func(loader_optional_nested), **kw)}
            return test_lambda(func(loader_optional_nested), **kw)

    result_i_d = {}
    for i, loader in enumerate(loaders):
        result_i_d.update(get_result(loader, i=test_names.get(i) or str(i), **kwargs))
    return NestedDict.from_dict(result_i_d)


def list_loader_fn(loaders, fn, **kwargs):
    if not isinstance(loaders, (list, tuple)):
        loaders = [loaders]
    return [(NestedDict.from_dict(loader) if isinstance(loader, dict) else loader).fn(fn, **kwargs) for loader in loaders]


def fit_shell(train_lambda, test_lambda, epochs, train_loader, test_loader_epoch,
              test_loader_final, test_names=None, loader_fn_kw=None, keepers=None,
              mode='fit', zero_result_kw=None,
              print_flag=True, epoch_print=True, print_0=False
              , _model=None, dl_cache=None, auto_caching=True, ensemble_caching=False,
              train_kw=None, test_kw=None, time_print=False, **kwargs):
    # if not print_flag:
    #     epoch_print = False
    print_0 = print_0 or print_flag or epoch_print
    if zero_result_kw is None:
        zero_result_kw = {}
    if test_names is None:
        test_names = ['test']
    test_names = dict(enumerate(test_names))
    if loader_fn_kw is None:
        loader_fn_kw = {}
    if keepers is None:
        keepers = KeeperDict()
    if train_kw is None:
        train_kw = {}
    if test_kw is None:
        test_kw = {}

    train_kw['prints'] = epoch_print

    frozen_ensemble_flag = _model is not None and _model.is_frozen_ensemble
    if frozen_ensemble_flag:
        ensemble_model = _model
        preproc_fn = ensemble_model.prepare_meta_dataloader

        def nested_preproc(x):
            return list_loader_fn(x, preproc_fn, update=False)
        # nested_preproc = lambda x: list_loader_fn(x, preproc_fn, update=False)
        if test_loader_epoch is not None:
            test_loader_epoch = nested_preproc(test_loader_epoch)
        # test_loader_epoch = _model.prepare_meta_dataloader(test_loader_epoch)
        model = ensemble_model
        lambda_kw = {'_model': ensemble_model.meta_model}
        test_lambda = ef.kw_wrapper(test_lambda, **lambda_kw)
        if mode == 'fit':
            train_lambda = ef.kw_wrapper(train_lambda, **lambda_kw, **train_kw)
            train_loader = preproc_fn(train_loader)

    if ensemble_caching:
        def trigger_preproc(_model, _test_dl, _dl_cache=None):
            if _model is not None and _model.is_frozen_ensemble:
                _ensemble_model = _model
                if _dl_cache is not None and 'test_dl' in _dl_cache and _dl_cache['test_dl'] is not None:
                    _test_dl = _dl_cache['test_dl']
                else:
                    _preproc_fn = _ensemble_model.prepare_meta_dataloader

                    def _nested_preproc(x):
                        return list_loader_fn(x, _preproc_fn, update=False)

                    if _dl_cache is None:
                        _dl_cache = {}
                    _dl_cache['test_dl'] = _nested_preproc(_test_dl)

                # _lambda_kw = {'_model': _ensemble_model.meta_model}
                # _test_lambda = ef.kw_wrapper(_test_lambda, **_lambda_kw)

                return _model, _test_dl
        trigger_inst = ef.FuncWithState(trigger_preproc, state_kwarg='_dl_cache')


    # def get_result(loaders, **kw):
    #     return _get_result_dict(loaders, test_lambda, **loader_fn_kw, **kw, **test_kw, **kwargs)
    if ensemble_caching:
        get_result = ef.FunctionArgBinder(get_result_dict, test_lambda, **loader_fn_kw, **test_kw, **kwargs,
                                          arg_changer_fn=trigger_inst, arg_changer_args_in=('model', 0))
    else:
        get_result = ef.FunctionArgBinder(get_result_dict, test_lambda, **loader_fn_kw, **test_kw, **kwargs)

    if mode == 'test' or mode == 'init_keepers':
        if dl_cache is not None and 'test_dl' in dl_cache and dl_cache['test_dl'] is not None:
            test_dl = dl_cache['test_dl']
        else:
            if test_loader_final is not None and frozen_ensemble_flag:
                test_loader_final = nested_preproc(test_loader_final)
            if test_loader_epoch is not None and test_loader_final is not None:
                test_dl =  test_loader_epoch + test_loader_final
            elif test_loader_epoch is not None:
                test_dl =  test_loader_epoch
            elif test_loader_final is not None:
                test_dl =  test_loader_final
            else:
                raise ValueError('No test data provided')
            if auto_caching and dl_cache is not None and not ('test_dl' in dl_cache and dl_cache['test_dl'] is None):
                dl_cache['test_dl'] = test_dl
        if mode == 'test':
            return get_result(test_dl, prints=bool(print_flag))

    results_epochs = NestedDictList()
    print(f"INIT [{0}]") if print_0 and mode != 'init_keepers' else None
    k_i_flag = mode == 'init_keepers'
    zero_result = get_result(test_dl if k_i_flag else test_loader_epoch, **zero_result_kw, prints=print_0, return_structure=k_i_flag or not print_0)
    keepers.build_nested(zero_result)

    # def lmbd_res(x):
    #     return get_result(test_dl, prints=False, _model=x)

    if mode == 'init_keepers':
        keepers.test_fn = ef.FunctionArgBinder(get_result, test_dl, prints=False, args2kwargs=('_model',))
        return keepers
    for epoch in range(1, epochs + 1):
        _epoch_print = isinstance(epoch_print, bool) and epoch_print or epoch_print is not None and epoch_print > 0 and epoch % epoch_print == 0 or epoch_print and epoch == 1
        print(f"Epoch [{epoch}]") if _epoch_print else print(f"Epoch [{epoch}...") if epoch==1 else None
        t = datetime.now()
        if frozen_ensemble_flag:
            train_lambda(train_loader)
        else:
            model = train_lambda(train_loader)
        if time_print: print(f"Time: {datetime.now() - t}")
        t = datetime.now()

        results_dict = get_result(test_loader_epoch, prints=_epoch_print)
        if time_print: print(f"Time: {datetime.now() - t}")
        t = datetime.now()
        model.config['epoch'] = epoch
        keepers.check_keep(model, results_dict, copy=True, add_tuples=(('epoch', epoch),))
        if time_print: print(f"Time: {datetime.now() - t}")
        results_epochs.append(results_dict)
        # print(f'BBB:{kb.agg_dict.ival(0).agg_dict}')
    # print(f"FINAL")
    # if frozen_ensemble_flag:
    #     test_loader_final = nested_preproc(test_loader_final)
    return results_epochs.concat()


def test_shell(data_loader_loop, metric=None, return_structure=False, name=None, prints=True):
    # prints = True
    if not isinstance(metric, (list, tuple)):
        metric = [metric]
    metric = ef.flatten(metric)
    metric = [m() if isinstance(m, MetricInstancer) else m for m in metric]
    metric = list(ef.flatten(metric))
    # print([p for p in model.parameters()])
    # print([p.grad for p in model.parameters()])
    loss_v = data_loader_loop(metric) if not return_structure else None
    if return_structure or loss_v is not None:
        out_l = {'loss': loss_v}
    else:
        out_l = {}

    if metric is not None:
        # print(f"Test Error: \n {metric('', '')}: {(metric):>0.1f}%, Avg loss: {loss:>8f} \n")
        out_t = out_l.copy()
        if return_structure:
            out_t |= {m.name: None for m in metric}
        else:
            out_t |= {m.name: m.compute(mean=0) for m in metric if not m.meta_metric}
            out_t |= {m.name: m.compute(metric_dict=out_t) for m in metric if m.meta_metric}
        out = out_l | {m.name: out_t[m.name] for m in metric}
    else:
        out = out_l
    out_nd = NestedDict.from_dict(out)
    if not return_structure:
        rounded = out_nd.fn(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, update=False).fn(ef.round_to, 6)
        if prints:
            if name is None:
                print(rounded)
            else:
                print(f'{name}: {rounded}')
    return out


def ts_make_ds(dict_split, inp_cols, tgt_cols, sequence_length, func=lambda x, y: (x, y), set_attr=None, return_ds=False):
    X_train, y_train = [], []
    for k_day_interval, v_day in dict_split.items():
        print(k_day_interval)
        _df = v_day.df
        _df_ind = v_day.df_index
        inp_df = _df.loc[_df_ind, inp_cols]
        tgt_df = _df.loc[_df_ind, tgt_cols]
        for i in range(len(_df_ind) - sequence_length):
            X_train.append(inp_df.iloc[i: i + sequence_length].to_numpy())
            y_train.append(tgt_df.iloc[i + sequence_length].to_numpy())
    train_ds = func(np.array(X_train), np.array(y_train))
    if set_attr is None:
        return train_ds
    else:
        setattr(dict_split, set_attr, train_ds)
        if return_ds:
            return train_ds


class MeanAbsoluteError:
    name = 'MAE'
    name2 = 'mae'
    alias = 'mean_absolute_error'

    def __call__(self, pred, y, mean=None, **kwargs):
        pred, y = self.transform_pred_y(pred, y, **kwargs)
        out = self.mean(abs(pred - y), mean=mean)
        self._post_call(out)
        return out


class MeanAbsoluteSignal:
    name = 'MAS'
    name2 = 'mas'
    alias = 'mean_absolute_signal'
    
    def __call__(self, pred, y, mean=None, **kwargs):
        pred, y = self.transform_pred_y(pred, y, **kwargs)
        out = self.mean(abs(y), mean=mean)
        self._post_call(out)
        return out


class MeanErrorRatio:
    name = 'MER'
    name2 = 'mer'
    alias = 'mean_error_ratio'
    meta_metric = True
    source_names = ['MAE', 'MAS']

    def __call__(self, metric_dict, **kwargs):
        out = metric_dict[self.source_names[0]] / metric_dict[self.source_names[1]]
        self._post_call(out)
        return out


class SignHitPercent:
    alias = 'sign_hit_percent'

    def __init__(self, zero_lim=0, **kwargs):
        if zero_lim is None:
            self.zth = 0
        else:
            self.zth = zero_lim
        if self.zth > 0:
            self.name = f'SHP_{round(ef.get_mantissa(self.zth))}_{ef.get_exponent(self.zth)}'
            self.name2 = 'shp_tr'
        else:
            self.name = 'SHP'
            self.name2 = 'shp'
        super().__init__(**kwargs)

    def __call__(self, pred, y, mean=None, **kwargs):
        # condition_conversion = SELF.check_get(condition_conversion, self)
        pred, y = self.transform_pred_y(pred, y, **kwargs)
        # Calculate the sign hit percentage
        condition = ((pred * y) > 0) | ((abs(pred) <= self.zth) & (abs(y) <= self.zth))
        # if condition_conversion is not None:
        #     condition = condition_conversion(condition, pred)
        out = self.mean(1.0 * condition, mean=mean)
        self._post_call(out)
        return out


class MeanOutputRatio:
    name = 'MOR'
    name2 = 'mor'
    alias = 'mean_output_ratio'

    def __call__(self, pred, y, mean=None, **kwargs):
        pred, y = self.transform_pred_y(pred, y, **kwargs)
        out =  self.mean(abs(pred), mean=mean) / self.mean(abs(y), mean=mean)
        self._post_call(out)
        return out
