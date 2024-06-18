import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

import operator
import itertools as it
import collections as cs



import m_objects as mo

import m_models as mm
import m_torch as mt
import m_sklearn as ms
import m_multiproc as mproc

import constants
import fit_utils as fu
import metrics_lib as mlb

import fin_dicts as fd
import fin_objects as fo
import extra_functions as ef


import m_db as mdb
import postgresql_db as pdb
import database_jobs as dbj

from torch.optim import lr_scheduler

from math import floor, ceil
from uuid import uuid4
from datetime import datetime, timedelta, date, time
from itertools import product
from m_objects import JSONEncoderCustom
from special_dicts import NestedDict, NestedDefaultDict, NestedDictList

from type_classes import *

import sklearn.ensemble as sk_e
import sklearn.linear_model as sk_lm
import sklearn.naive_bayes as sk_nb
import sklearn.svm as sk_svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


sql_conn = mdb.create_db_connection()
sql_engine = mdb.create_db_engine()

def dumps(j): return json.dumps(j, sort_keys=True, default=str)

# Configure run
LIB = 'sklearn'
LIB = 'torch'
DEVICE = mt.get_default_device('GPU')
UPLOAD = True
COMMIT = False
stock_schema = 'trades'
models_schema = 'dm'

DEBUG = 1

SHUFFLE_TRAIN = True

# Stock data
tickers = ['AAPL', 'FXI', 'SPY']
# tickers = ['SPY']

tickers_in = []
tickers_target = ['SPY']

# barSizesSamples = {'1 month': 5 , '1 week': 5 , '1 day': 5}
bar_sizes = ['1m', '1D']
inputColumns = ['percent']
targetColumns = ['percent']
inputColumns = ['percent', 'percent_high-low', 'volume_pct', 'low_pct', 'high_pct', 'average_pct']
a_c = ['percent', 'percent_high-low', 'volume_pct', 'low_pct', 'high_pct', 'average_pct', 'close_pct']
b_c = ['open_pct_D', 'close_pct_D', 'low_pct_D', 'high_pct_D', 'average_pct_D']
inputColumns = a_c + b_c
# inputColumns = ['percent']
# targetColumns = ['percent']
startDate = datetime(2023,10,1)
startDate = datetime(2015,1,1)
endDate = None
endDate = datetime(2015,6,30)
index = 'date'


DATA_OPS = lambda: [mask([us_t.market_hours]), split_y(['1Y']), split_y(['1M']), split(['1D'])]
N_masks = 1

if DEBUG:
    inputColumns = ['percent', 'volume_pct']
    inputColumns = ['percent', 'percent_high-low', 'volume_pct', 'low_pct', 'high_pct', 'average_pct']
    a_c = ['percent', 'percent_high-low', 'volume_pct', 'low_pct', 'high_pct', 'average_pct', 'close_pct']
    # a_c = ['percent', 'percent_high-low', 'volume_pct', 'low_pct', 'high_pct', 'average_pct']
    b_c = ['open_pct_D', 'close_pct_D', 'low_pct_D', 'high_pct_D', 'average_pct_D']
    # inputColumns = ['percent', 'open_pct_D', 'open_pct_W', 'close_pct_D', 'close_pct_W', 'low_pct_D', 'low_pct_W', 'high_pct_D', 'high_pct_W']
    # inputColumns = ['open_pct_D', 'close_pct_D', 'low_pct_D', 'high_pct_D', 'average_pct_D', 'close_pct_D']
    inputColumns = a_c + b_c
    # inputColumns = a_c
    # inputColumns = inputColumns + dbj.CREATE_PREVIOUS_DATE_RATIO_COLUMNS_columns
    startDate = datetime(2015, 1, 1)
    endDate = datetime(2015, 2, 28)


# Training params
# testFrac = 0.3
TORCH_DTYPE = torch.float
RANDOM_STATE = constants.random_state
SEQUENCE_LENGTH = 30

LOSS = nn.MSELoss()

if any([tickers_in, tickers_target]):
    tickers = ef.unique(tickers_in + tickers_target)
tickers_in = tickers_in if tickers_in else tickers
tickers_target = tickers_target if tickers_target else tickers

TO_DB_TABLE = {'tickers_in': tickers_in, 'tickers_target': tickers_target,
               'bar_sizes_in': bar_sizes, 'bar_sizes_target': bar_sizes, 'input_columns': inputColumns,
               'target_columns': targetColumns, 'library': LIB, }

if LIB == 'torch':
    LIB_SRC = mt
elif LIB == 'sklearn':
    LIB_SRC = ms

# Fetch or build data
BS_TICKER_DFS = fd.FinDict()
t = datetime.now()
print('STARTING DB DOWNLOAD of ', tickers, ' at ', bar_sizes, 'bar sizes between ', startDate, ' and ', endDate)
for bs in bar_sizes:
    def base_f(x):
        return pdb.get_table_as_df(sql_engine, [stock_schema, f'{x}_{bs}'], index=index,
                                   between=(startDate - timedelta(days=365) if 'D' in bs else startDate , endDate))
    def concur_func(x, d):
        d[x] = base_f(x)
    parallel_func = mproc.thread_concurrently
    # parallel_func = mproc.mp_concurrently
    ticker_dfs = fd.FinDict(parallel_func(concur_func, tickers, results={}))
    BS_TICKER_DFS[bs] = ticker_dfs
print(f'FINISHED IN {datetime.now() - t}')

BS_TICKER_DFS.fn(lambda _df: _df.set_index(ef.to_timezone(_df.index, 'America/New_York')))

# inputColumns += fo.add_sma_columns(BS_TICKER_DFS, ['open', 'close', 'high', 'low'], [5, 10, 20, 50, 100, 200])
# inputColumns += fo.add_ta_columns(BS_TICKER_DFS)

us_t = fo.USTimes
mask = lambda x, **kw: lambda d: fo.dict_multi_time_mask(d, x, **kw)
split = lambda x, **kw: lambda d: fo.dict_freq_split(d, x, **kw)
split_y = lambda x, **kw: lambda d: fo.dict_freq_split(d, x, save_split_df=True, append_small_splits=True, **kw)

# op = [mask([us.extended_hours_long, us.market_hours, us.extended_hours, us.open_noon, us.open35]), split(['1D'])]

data_ops = DATA_OPS()
BS_TICKER_DFS.fn(ef.join_dataframes_on_index, level=1)
BS_TICKER_DFS.fn(lambda _df: fd.TimeSeriesDict.from_df(_df, data_ops))
BS_TICKER_DFS.set_links()

inp_cols = []
[inp_cols.append(f'{t}_{c}') for c in inputColumns for t in tickers]
tgt_cols = []
[tgt_cols.append(f'{t}_{c}') for c in targetColumns for t in tickers]



def create_fit_fn(sequence_length=SEQUENCE_LENGTH, db_upload_dict=None, random_state=RANDOM_STATE):
    level = 2 * (len(data_ops)-N_masks) + N_masks - 1
    if db_upload_dict is not None:
        db_upload_dict |= {'sequence_length': sequence_length}
    d_1m_data = BS_TICKER_DFS['1m']
    d_1m = d_1m_data.fn(LIB_SRC.ts_make_ds, inp_cols, tgt_cols, sequence_length, set_attr='ds', level=level, dtype=TORCH_DTYPE)
    def interval_key_fn(x):
        if isinstance(x, pd.Interval):
            ts = x.left
            match ef.td_str(x.right - x.left, time_quants=['Y', 'M', 'D'], no_prefix=True, r=0.1):
                case 'Y':
                    return f'{ts.year}'
                case 'M':
                    return f'{ts.month}'
        return x

    d_1m.keys_func(interval_key_fn).set_links()
    tmp = LIB_SRC.nested_helper_func(debug=DEBUG, level=level, accessor=lambda x: x.ds, random_state=random_state)
    construct_test_f_fn, construct_dl_fn, construct_agg_fn = tmp
    test_f_fn = construct_test_f_fn(d_1m)
    dl_fn = construct_dl_fn()
    # test_short = test_f_fn(0.1)
    # test_short_agg_fn = construct_agg_fn(test_short)
    # test_short_agg_dl_fn = lambda **kw: dl_fn(test_short_agg_fn(**kw))
    # test_short_dl_fn = lambda **kw: dl_fn(test_short, **kw)
    short_frac = 1. if LIB == 'torch' else 1.
    long_frac = 1. if LIB == 'torch' else 1.
    test_short_dl_tup = dl_fn(test_f_fn(short_frac))
    test_long_dl_tup = dl_fn(test_f_fn(long_frac))

    test_short_dl= test_short_dl_tup.fn(lambda x: x[0], update=False)
    test_long_dl = test_long_dl_tup.fn(lambda x: x[0], update=False)
    test_short_dl_rest = test_short_dl_tup.fn(lambda x: x[1], update=False) if short_frac < 1. else None

    # keeper_names = ['bk_loss', 'bk_loss_mor_high', 'bk_loss_mor_low', 'bk_mer_top_mor_high', 'bk_shp_top_mor_high',
    #                 'bk_shp_tr_mor_high', 'bk_shp_tr_shp_top_mor_high', 'bk_shp_tr_shp_top_mor_low']
    # keeper_names = ['bk_shp_tr_shp_top_mor_low'] if DEBUG else keeper_names
    metrics_ls, keepers_dict = mlb.generate_kp_and_best_keepers(LIB_SRC, tgt_cols, metrics=None)

    fn_dict_fit = lambda **kw: d_1m.fn(fit_upload, test_dl_epoch=test_short_dl, test_dl_final=test_long_dl, test_dl_rest=test_short_dl_rest,
                                       metrics=metrics_ls, keepers=keepers_dict, update=False, level=level, epoch_test_different=short_frac!=long_frac, **kw)
    return fn_dict_fit


def upload_model(model, db_package, prime=False):
    to_binary = {'model': model} if not prime else {}
    to_table = {**TO_DB_TABLE, **db_package,
                'index': index, 'binary': LIB_SRC.dump_bytes_io(to_binary).read(),
                'binary_keys': list(to_binary.keys()).sort(), 'text': fu.to_str(model),
                'text_long': str(to_binary).replace('\n', '')}
    cmp_insert = pdb.composed_insert([models_schema, 'models' + '_prime' if prime else ''], to_table.keys(), returning=['uuid'])
    dm_uuid, = pdb.pg_execute(sql_conn, cmp_insert, list(to_table.values()), commit=COMMIT, as_string=True)[0]
    return dm_uuid

def fit_upload(dict_split, nesting, /, train=True, test_dl_epoch=None, iter_times=1,
               model_cls=mm.LSTMModel, model_params=None, model_iter_options=None,
               ensemble_cls=mm.Ensemble, ensemble_params=None, ensemble_iter_options=None,
               test_dl_final=None, test_dl_rest=None,
               epochs=1, batch_size=None,
               loss_fn=LOSS, scaler_cls=None,
               optimizer_fn=None, optimizer_params=None, scheduler_fn=None, scheduler_params=None,
               device=DEVICE,
               metrics=None, shuffle_train=SHUFFLE_TRAIN, lib=LIB, keepers=None,
               db_upload_dict=None, epoch_print=False, print_less=True,
               epoch_test_different=True,
               print_only_mean=False,
               sequence_length=SEQUENCE_LENGTH, **kwargs):

    if db_upload_dict is None:
        db_upload_dict = {}
    if model_params is None:
        model_params = {}
    if ensemble_params is None:
        ensemble_params = {}
    elif not isinstance(ensemble_params, (list, tuple)):
        ensemble_params = dict(ensemble_params)
    if optimizer_params is None:
        optimizer_params = {}

    if not isinstance(test_dl_epoch, (list, tuple)):
        test_dl_epoch = [test_dl_epoch]
    if not isinstance(test_dl_final, (list, tuple)):
        test_dl_final = [test_dl_final]

    loader_fn_kw = {'ignore_keys': [dict_split.nesting[-2]]}
    loader_fn_kw = {}
    tests_dls = test_dl_epoch, test_dl_final, test_dl_rest
    tests_dls = [[tst] if not callable(tst) and not isinstance(tst, (list, tuple)) else tst for tst in tests_dls]
    test_dl_epoch, test_dl_final, test_dl_rest = [[x(**loader_fn_kw) if callable(x) else x for x in tst] for tst in tests_dls]
    input_size = len(inp_cols)
    output_size = len(tgt_cols)
    fit_params_2db = db_upload_dict | {'shuffle_train': shuffle_train, 'epochs': epochs, 'batch_size': batch_size}
    model_params_2db = {'scaler': None if scaler_cls is None else str(scaler_cls)}

    common_fit_kw = dict(epoch_print=epoch_print, device=device, print_flag=not print_less)

    t_i_keys_fn = lambda x: x.keys_func(lambda y: str(y) if isinstance(y, mo.TimeInterval) else y)
    _y_f = lambda _: str(ef.round_to(_, 5)).ljust(len(_) * 11 + 2 if isinstance(_, Iterable) else 11)
    preprint_fn = lambda x: t_i_keys_fn(x).fn(lambda y, nes,/: _y_f(y) if nes[-1][:3] != 'MAS' else y, update=False)
    res_list_print_fn = lambda x, add_before='': print(add_before, preprint_fn(x))
    if print_less or print_only_mean:
        agg_res_list_print_fn = lambda x, add_before='': print(add_before, preprint_fn(x).change_nesting(inner_out=1)['mean'])
    else:
        agg_res_list_print_fn = res_list_print_fn
    keeper_print_shell = lambda print_fn: lambda x: print_fn(x, add_before=f'keeper {str(x.name).upper().ljust(33)}:')

    def test_fn(model_tests, _name=None):
        model_tests_copy = model_tests.copy().fn('del_n', nesting[:-1], level=0)
        model_tests_agg = model_tests_copy.func(fu.agg_nested)
        print('AGGREGATED STATS:')
        model_tests_agg.func(agg_res_list_print_fn)

    def local_fit_shell(fit_fn, _keepers, name, create_keepers=True, cache_first_dl=False, test=True):
        if create_keepers:
            _keepers = _keepers.create(name='ds_epoch')
        t = datetime.now()
        results_epochs = fit_fn(_keepers)
        trained_models = NestedDictList(_keepers.get_models())
        print(F'FINISHED {name} TRAINING AND KEPT {len(trained_models)} BEST MODELS OUT OF', epochs, 'EPOCHS IN', datetime.now() - t)

        model_test_f_loader = lambda loader, dl_cache: lambda x: LIB_SRC.fit(x, None, None, None, loader, mode='test',
                                   loader_fn_kw=loader_fn_kw if not DEBUG else None, loss=loss_fn,
                                   metric=metrics, **common_fit_kw, dl_cache=dl_cache if cache_first_dl else None)
        model_test_f_epoch = model_test_f_loader(test_dl_epoch, {})
        model_test_f_final = model_test_f_loader(test_dl_final, {})


        def _test_fn(_test_f, _name=None):
            print(_name)
            t = datetime.now()
            model_tests = trained_models.func(_test_f)
            test_fn(model_tests, _name=_name)
            print(f'{_name} DONE IN', datetime.now() - t)

        if test:
            if epoch_print and epoch_test_different:
                _test_fn(model_test_f_epoch, _name='TEST_DL_EPOCHS')
            _test_fn(model_test_f_final, _name='TEST_DL_FINAL')
        return trained_models, _keepers, results_epochs


    def init_keepers_with_test(_keepers, test_dl, cache_first_dl=False, name=None):
        _keepers = _keepers.create(name=name)
        LIB_SRC.fit(None, None, None, None, test_dl, mode='init_keepers', metric=metrics, keepers=_keepers,
                    loader_fn_kw=loader_fn_kw if not DEBUG else None, loss=loss_fn, dl_cache={} if cache_first_dl else None, **common_fit_kw)
        return _keepers


    def fit_over_iter_options(_model_cls, _dict_split, _test_dl_epoch, _test_dl_final, _test_dl_rest, keeper_epoch, keeper_rest,
                              model_args=(input_size, output_size), _model_params=None, _iter_options=None, fit_shell_kw=None, split_dl_epoch=True, ):

        if fit_shell_kw is None:
            fit_shell_kw = {}
        _test_dl_epoch_const = _test_dl_epoch
        # _test_dl_rest_const = _test_dl_rest
        _device = device
        _loss_fn = loss_fn
        is_none_fn = lambda x: x is None or isinstance(x, (list, tuple)) and all([y is None for y in x])
        if _model_params is None:
            _model_params = {}


        init_keeper_fn = lambda _keeper, _dl, **kw: init_keepers_with_test(_keeper, _dl, **kw) # , cache_first_dl=fit_shell_kw.get('cache_first_dl', False)
        keep_models_fn = lambda _keeper, _dl, model_source, **kw: init_keeper_fn(_keeper, _dl, **kw).test_check_keep(model_source)
        get_models_fn = lambda _keeper, _dl, model_source: keep_models_fn(_keeper, _dl, trained_models).get_models()

        def keeper_coerce(x):
            if isinstance(x, tuple):
                k_cls = keepers[x[0]].__class__
                if issubclass(k_cls, dict):
                    return k_cls({k: keepers[k] for k in x if k in keepers})
                elif issubclass(k_cls, list):
                    return k_cls([keepers[k] for k in x])
                return k_cls([keepers[k] for k in x])
            elif isinstance(x, str):
                return keepers[x]
            return x

        keeper_epoch = keeper_coerce(keeper_epoch)
        keeper_rest = keeper_coerce(keeper_rest)

        dict_init = lambda: NestedDefaultDict()
        list_init = lambda: NestedDictList()
        def_dict_init = lambda x: NestedDefaultDict(list_init, levels=x)

        model_fit_out = dict_init()
        model_fit_out['models'] = def_dict_init(2)
        model_fit_out['results'] = def_dict_init(2)
        iterator = [{}] if not _iter_options else _iter_options
        for iter_op_i, _iteration_options in enumerate(iterator):
            it_o = _iteration_options
            _model_cls = it_o.get('model_cls', _model_cls)
            _optimizer_fn = it_o.get('optimizer_fn', optimizer_fn)
            _optimizer_params = it_o.get('optimizer_params', optimizer_params)
            _scheduler_fn = it_o.get('scheduler_fn', scheduler_fn)
            _scheduler_params = it_o.get('scheduler_params', scheduler_params)
            _scheduler_set = it_o.get('scheduler_set')
            _params_dict = {'model_cls': _model_cls, 'optimizer_fn': _optimizer_fn, 'optimizer_params': _optimizer_params,
                            'scheduler_fn': _scheduler_fn, 'scheduler_params': _scheduler_params, 'scheduler_set': _scheduler_set}

            model_params_iter = ef.update_dict(_model_params.copy(), it_o)
            if 'seq_len' in _model_cls.parameters_names:
                model_params_iter['seq_len'] = sequence_length

            if 'model_params' in it_o:
                model_params_iter |= it_o['model_params']

            cls_name = _model_cls.name
            if cls_name not in model_fit_out['keepers']:
                model_fit_out['keepers'][cls_name]['k_epoch_dl_final'] = init_keeper_fn(keeper_epoch, _test_dl_final, name='ds_final')
                # model_fit_out['keepers'][cls_name]['rest_final'] = init_keepers_with_test(keeper_rest, _test_dl_final)
                if any(_test_dl_rest):
                    model_fit_out['keepers'][cls_name]['k_rest_dl_rest'] = init_keeper_fn(keeper_rest, _test_dl_final, name='ds_final')

            for i_t in range(iter_times):
                print(f'ITERATION OPTIONS {iter_op_i+1} out of {len(iterator)}')
                print('ITERATION OPTIONS:', it_o)
                print(f'iteration {i_t+1} out of {iter_times}')
                if lib == 'torch':
                    split_fn = lambda x, f=0.3: tud.random_split(x, [f, 1-f], generator=None) # .manual_seed(RANDOM_STATE)
                    dl_fn = lambda x, shuffle=False: tud.DataLoader(x, batch_size, shuffle=shuffle)
                    trainDL = tud.DataLoader(_dict_split.ds, batch_size, shuffle=shuffle_train)
                    # trainDL = dl_fn(split_fn(_dict_split.ds, f=0.5)[0], shuffle=shuffle_train)
                    train_test_dl = dl_fn(split_fn(_dict_split.ds)[0])

                    if split_dl_epoch:
                        def nest_fn(x, fn, split_nested=False):
                            if isinstance(x, list):
                                z = map(lambda y: nest_fn(y, fn, split_nested=split_nested), x)
                                if not split_nested:
                                    return list(z)
                                return tuple(map(list, zip(*z)))
                            elif isinstance(x, NestedDict):
                                z = x.fn(fn, update=False)
                                if not split_nested:
                                    return z
                                return z.fn(lambda y: y[0], update=False), z.fn(lambda y: y[1], update=False)
                            else:
                                return fn(x)
                        _test_dl_epoch, _test_dl_epoch_split = nest_fn(_test_dl_epoch_const, lambda x: split_fn(x.dataset, f=0.5), split_nested=True)
                        _test_dl_epoch, _test_dl_epoch_split = nest_fn([_test_dl_epoch, _test_dl_epoch_split], dl_fn)
                    config_dict = NestedDict.from_dict(dict(filtered=_params_dict, input=it_o)).fn(str).to_dict()
                    config_dict |= {'train_ds': nesting}
                    model = _model_cls(*model_args, **model_params_iter, config=config_dict).to(_device)
                    optimizer = _optimizer_fn(model.parameters(), **_optimizer_params)
                    scheduler = None if _scheduler_fn is None else _scheduler_fn(optimizer, **_scheduler_params)
                    if _scheduler_set is not None:
                        [setattr(scheduler, k, v) for k, v in _scheduler_set.items()]
                    optimizer_param = optimizer.param_groups[0].copy()
                    optimizer_param.pop('params')
                    # _model_params_2db = model_params_2db | model.get_name_and_parameters() | {'optimizer': optimizer_param}
                elif lib == 'sklearn':
                    trainDL = _dict_split.ds
                    train_test_dl = LIB_SRC.split(_dict_split.ds, 0.3, random_state = RANDOM_STATE)
                    _model = _model_cls(**_model_params)
                    # _model_params_2db = model_params_2db | {'model': str(_model_cls)} | _model.get_params()
                    if scaler_cls is None:
                        model = _model
                    else:
                        model = Pipeline([('scaler', scaler_cls()), ('model', _model_cls(**model_params_iter))])
                    #
                    optimizer = None
                    scheduler = None
                    _device = None
                    _loss_fn = None
                if not isinstance(_test_dl_epoch, list):
                    _test_dl_epoch = [_test_dl_epoch]
                test_dl_epoch2 = [{'val': train_test_dl}] + _test_dl_epoch
                print('MODEL:', model, 'LOSS:', _loss_fn, 'OPTIMIZER:', '' if print_less else optimizer, '\n',
                      'TRAIN RANGE:  [', _dict_split.t[0], '-', _dict_split.t[1], ']\n')
                fit_fn = lambda ke: LIB_SRC.fit(model, epochs, trainDL, test_dl_epoch2, _test_dl_final,
                                                loader_fn_kw=loader_fn_kw if not DEBUG else None, optimizer=optimizer,
                                                scheduler=scheduler, loss=_loss_fn, metric=metrics, keepers=ke, **common_fit_kw)

                trained_models, model_keepers, results_epoch  = local_fit_shell(fit_fn, keeper_epoch, _model_cls.name, test=False, **fit_shell_kw)

                t = datetime.now()
                model_fit_out['models'][cls_name]['epoch'].append(trained_models)
                model_fit_out['results']['epoch'][model.uuid].append(trained_models)
                # if split_dl_epoch:
                #     model_fit_out['models'][cls_name]['split'].append(get_models_fn(keeper_rest, _test_dl_epoch_split, trained_models))
                # if any(_test_dl_rest):
                #     model_fit_out['models'][cls_name]['rest'].append(get_models_fn(keeper_rest, _test_dl_rest, trained_models))
                print(f'Starting test_check_keep for {cls_name}')
                test_fn(model_fit_out['keepers'][cls_name]['k_epoch_dl_final'].test_check_keep(trained_models, return_results=True), _name='TEST_DL_FINAL')
                print(f'FINISHED test_check_keep for {cls_name} in {datetime.now() - t}')

        model_fit_out = NestedDict(model_fit_out)
        model_fit_out['models'] = model_fit_out['models'].cast(NestedDict)
        model_fit_out['keepers'] = model_fit_out['keepers'].cast(fu.KeeperDict)
        return model_fit_out


    def fit_models(_model_cls=model_cls, _dict_split=dict_split, _test_dl_epoch=test_dl_epoch, _test_dl_final=test_dl_final,
                   _test_dl_rest=test_dl_rest, keeper_fit=('k1', 'k_shp_mer'), keeper_rest=('k2', 'k_shp_mer'), model_args=(input_size, output_size),
                   _model_params= model_params, _iter_options=model_iter_options):
        return fit_over_iter_options(_model_cls, _dict_split, _test_dl_epoch, _test_dl_final, _test_dl_rest, keeper_fit,
                                     keeper_rest, model_args=model_args, _model_params=_model_params,_iter_options=_iter_options)


    def _fit_ensembles(_model_cls=ensemble_cls, _dict_split=dict_split, _test_dl_epoch=test_dl_epoch, _test_dl_final=test_dl_final,
                       _test_dl_rest=test_dl_rest, keeper_fit=('k2', 'k_shp_mer'), keeper_rest=('k3', 'k_shp_mer'), model_args=None,
                       _model_params=ensemble_params, _iter_options=ensemble_iter_options):
        if _model_params is None:
            _model_params = {}
        if 'is_frozen_ensemble' not in _model_params:
            _model_params['is_frozen_ensemble'] = True
        return fit_over_iter_options(_model_cls, _dict_split, _test_dl_epoch, _test_dl_final, _test_dl_rest, keeper_fit,
                                     keeper_rest, model_args=model_args, _model_params=_model_params,
                                     _iter_options=_iter_options, fit_shell_kw={'cache_first_dl': True})

    def fit_ensembles(model_ls, name=None, **kw):
        if name is not None:
            print('Fitting ensemble for', name, f' with {len(model_ls)} models', )
        return _fit_ensembles(model_args=(model_ls,), **kw)

    if not train:
        return fit_models, fit_ensembles

    k_get_models = lambda x: ef.flatten([y.get_models() for y in ef.flatten(x, type_=tuple) ], type_=list)

    if DEBUG and DEBUG == 2:
        fit_out = ef.pickle_load('keepers4ensemble')
    else:
        fit_out = fit_models()
        if not DEBUG==1:
            ef.pickle(fit_out, 'keepers4ensemble')
    # 'k_shp_mer'
    final_src  = keepers.__class__(fit_out['keepers'].copy()).change_nesting(pop_insert=[0, 1])['k_epoch_dl_final'].change_nesting(pop_insert=[0, 1])['k_shp_mer']
    cls_inner_src = final_src.copy().change_nesting(outer_in=1)


    # cnnt_src = keepers_src['CNNTimeSeries']
    # transformer_src = keepers_src['Transformer']

    def get_src_and_ex(name='LSTM'):
        # if name not in final_src:
        #     return None, None
        model_src = final_src[name]
        final_src_ex_lstm = final_src.copy().del_n(name) if name in final_src else final_src.copy()
        cls_ex_src = final_src_ex_lstm.copy().change_nesting(outer_in=1)
        return model_src, cls_ex_src
    #
    # lstm_src, cls_ex_lstm_src = get_src_and_ex('LSTM')
    #
    # lstm_reg = lstm_src['reg_low'], lstm_src['reg_mid']
    # ex_top_mid = cls_ex_lstm_src['top_mid']
    # ens_fit_lstm_reg_ex_top_mid = fit_ensembles(k_get_models((lstm_reg, ex_top_mid)), name='lstm_reg_ex_top_mid')
    # ens_fit_lstm_reg_all_top_mid = fit_ensembles(k_get_models((lstm_reg, all_top_mid)), name='lstm_reg_all_top_mid')


    all_reg = cls_inner_src['reg_low'], cls_inner_src['reg_mid']

    all_top_mid = cls_inner_src['top_mid']




    ens_fit_all_reg_all_top_mid = fit_ensembles(k_get_models((all_reg, all_top_mid)), name='all_reg_all_top_mid')

    555
    # keepers_k2 = init_keepers_with_test(keepers['k2'], test_dl_final)
    # keepers_k3 = init_keepers_with_test(keepers['k3'], test_dl_final)
    # keepers_k4 = init_keepers_with_test(keepers['k_shp_top'], test_dl_final)
    # get_flat_models = lambda x: x.flatten(use_inner_cls=True).get_models()
    # # get_keeper_models = lambda _keeper, _models: _keeper.test_check_keep(_models).
    #
    # fitted_ensemble_m1k1 = [fit_ensembles(model_list) if model_list else {} for model_list in models_1_k1]
    # fitted_ensemble_m1k1.append(fit_ensembles(ef.flatten(models_1_k1)))
    # fitted_ensemble_m1k2 = [fit_ensembles(model_list) if model_list else {} for model_list in models_1_k2]
    # fitted_ensemble_m1k2.append(fit_ensembles(ef.flatten(models_1_k2)))
    #
    # # for model_list_lists in model_fit_out['models']:
    # #     for model_list in model_list_lists:
    # #         ensemble_fit_out = fit_ensembles(model_list)
    # # final_keepers_epoch = []
    # final_keepers = init_keepers_with_test(keepers['k2'], test_dl_final)
    # keepers_k2.test_check_keep(get_flat_models(models_1_k1)).test_check_keep(get_flat_models(models_1_k2))
    # keepers_k3.test_check_keep(get_flat_models(models_1_k1)).test_check_keep(get_flat_models(models_1_k2))
    # keepers_k4.test_check_keep(get_flat_models(models_1_k1)).test_check_keep(get_flat_models(models_1_k2))
    # # for model_list_lists in model_fit_out['models']:
    # #     for model_list in model_list_lists:
    # #         ensemble_fit_out = fit_ensembles(model_list)
    # # final_keepers_epoch = []
    # final_keepers = init_keepers_with_test(keepers['k2'], test_dl_final)
    # final_keepers.test_check_keep(model_keepers_rest.flatten(use_inner_cls=True).get_models())
    #
    # for ens_dict in fitted_ensemble_dicts:
    #     if not ens_dict:
    #         continue
    #     for model_list_lists in ens_dict['models']:
    #         # final_keepers_epoch.append(init_keepers_with_test(keepers['k1'], test_dl_epoch))
    #         # final_keepers_rest.append(init_keepers_with_test(keepers['k2'], test_dl_rest))
    #         for model_list in model_list_lists:
    #             # final_keepers_epoch[-1].test_check_keep(model_list)
    #             final_keepers.test_check_keep(model_list)
    #
    # return final_keepers.get_models()


    # for fit_dict in [model_fit_out, ensemble_fit_out]:
    #     for model_list_lists in fit_dict['models']:
    #         # final_keepers_epoch.append(init_keepers_with_test(keepers['k1'], test_dl_epoch))
    #         # final_keepers_rest.append(init_keepers_with_test(keepers['k2'], test_dl_rest))
    #         for model_list in model_list_lists:
    #             # final_keepers_epoch[-1].test_check_keep(model_list)
    #             final_keepers_rest.test_check_keep(model_list)
    #
    # keepers_ens
    # trained_ensembles = NestedDictList(keepers_ens.get_models())
    # lambda_return = lambda: (trained_models, model_tests_agg, model_tests)
    # if not UPLOAD:
    #     return lambda_return()
    #
    # js_dump = lambda x: json.dumps(x, cls=JamesonEncoder)
    # keeper_stats = {'agg': keepers_m.best_agg(), 'best': keepers_m.get_best()}
    # model_stats = NestedDict({'model_tests_agg': model_tests_agg, 'model_tests': model_tests})
    #
    # db_package =  {'json': js_dump(dict(model_params=model_params_2db, fit_params=fit_params_2db)),
    #                'start_date': dict_split.t[0], 'end_date': dict_split.t[1]}
    # db_package_prime =  {'keeper_stats': js_dump(keeper_stats), 'model_tests': js_dump(model_stats)}
    #
    # upload_model(trained_models[0], {**db_package, **db_package_prime,}, prime=True)
    #
    # _upload_model = lambda i: upload_model(trained_models[i],
    #                                        {**db_package, 'model_tests': {
    #                                            'model_tests_agg': model_tests_agg[i], 'model_tests': model_tests[i]}})
    # map(_upload_model, range(len(trained_models)))
    # return model, tests
    # return lambda_return()

list_fn = lambda name, vals: list(map(lambda y: {name: y}, vals))
permute_fn = lambda *ar: list(map(lambda x: dict(cs.ChainMap(*x)), it.product(*ar)))
meta_kw_fn = lambda dicts, meta: list(map(lambda y: {meta: y}, dicts))

fit = create_fit_fn()

skl_kw = dict(scaler_cls=StandardScaler)

gen_d =  {'random_state': RANDOM_STATE, 'verbose': 0}

trees_d = {'n_estimators': 100, **gen_d}
rf_r = dict(model_cls=sk_e.RandomForestRegressor, model_params=trees_d | {'n_jobs': -1})
rf_c = dict(model_cls=sk_e.RandomForestClassifier, model_params=trees_d | {'n_jobs': -1})
gb_r = dict(model_cls=sk_e.GradientBoostingRegressor, model_params=trees_d | {})

gb_hr = dict(model_cls=sk_e.HistGradientBoostingRegressor, model_params=gen_d | {})

gb_c = dict(model_cls=sk_e.GradientBoostingClassifier, model_params=trees_d)
sgdc = dict(model_cls=sk_lm.SGDClassifier)
lr = dict(model_cls=sk_lm.LogisticRegression)
gnb = dict(model_cls=sk_nb.GaussianNB)
mnb = dict(model_cls=sk_nb.MultinomialNB)
svc = dict(model_cls=sk_svm.SVC)

skl_kw |= gb_hr

batch_size_train = 32

epochs_train = 1000 if not DEBUG==1 else 200

iter_train = 3 if not DEBUG==1 else 1
max_lr = 1e-3 if not DEBUG==1 else 5e-3


torch_kw = dict()

lstm_model = dict(model_cls=mm.LSTMModel, model_params={'hidden_size': 64, 'num_layers': 3, 'lstm_init': 1, 'num_heads': 8})
transformer_model = dict(model_cls=mm.TransformerTimeSeries)
transformer_drop_0 =  transformer_model.copy() | dict(model_params=dict(dropout=0.0))
transformer_drop =  transformer_model.copy() | dict(model_params=dict(dropout=0.1))
cnnt_model = dict(model_cls=mm.CNNTimeSeries)
cnnt_permute = cnnt_model.copy() | dict(model_params=dict(permute=True))
adv_cnnt_model = dict(model_cls=mm.AdvancedCNNTimeSeries)

cnnt_lstm_model = dict(model_cls=mm.CNNLSTMModel)
# models_kw = [lstm_model, transformer_base, cnnt_model]
models_kw = [cnnt_permute]
models_kw = [adv_cnnt_model]
models_kw = [lstm_model]
# models_kw = [lstm_model, cnnt_model]
# models_kw = [transformer_base]
# models_kw = [transformer_drop_0, transformer_drop]


optimizer_p = {'lr': max_lr}
adam_kw = dict(optimizer_fn=optim.Adam, optimizer_params={} | optimizer_p)
sgd_kw = dict(optimizer_fn=optim.SGD, optimizer_params={} | optimizer_p)
optimizer_kw = [sgd_kw, adam_kw]
optimizer_kw = [adam_kw]

step_lr_kw = dict(scheduler_fn=lr_scheduler.StepLR, scheduler_params={'step_size': 40, 'gamma': 0.666})
one_cyc_p = dict(max_lr=max_lr, epochs=epochs_train, steps_per_epoch=ceil(7200/batch_size_train), base_momentum=0.90, max_momentum=0.95, div_factor=3000., final_div_factor=1e2)
one_cyc_p = dict(max_lr=max_lr, epochs=epochs_train, steps_per_epoch=ceil(7200/batch_size_train))
one_cycle_lr_kw = dict(scheduler_fn=optim.lr_scheduler.OneCycleLR, scheduler_params=one_cyc_p, scheduler_set=dict(batch_step=True))
scheduler_kw = [one_cycle_lr_kw, step_lr_kw]
# scheduler_kw = [step_lr_kw]

model_iterate = permute_fn(models_kw, optimizer_kw, scheduler_kw)

# ensemble_kw0 = dict(ensemble_cls=mm.Ensemble)
ensemble_kw = dict(ensemble_cls=mm.Ensemble, ensemble_params={'is_frozen_ensemble': True, 'meta_model': mm.MetaModel})
ensemble_kw = dict(ensemble_cls=mm.Ensemble, ensemble_params={'is_frozen_ensemble': True, 'meta_model': mm.AdvancedMLP1})
# ensemble_kw = dict(ensemble_cls=mm.Ensemble, ensemble_params={'is_frozen_ensemble': True, 'meta_model': mm.CapsuleMLP})


# ensemble_iterate = permute_fn('hidden_size', [8, 12, 16, 24, 32])
ens_iterate_hidden_sizes = list_fn('hidden_sizes', [(4, 2), (8, 4), (16, 8), (32, 16)])
ens_iterate_hidden_sizes = list_fn('hidden_sizes', [(16, 8), (32, 16)])
ens_iterate_nas = [{}, {'use_attention': True}, {'use_residual': True}, {'use_attention':True,'use_residual':True}]
ens_iterate_nas = [{}, {'use_attention':True, 'use_residual':True}]

# ens_iterate_nas = [{'use_residual': True}, {'use_attention':True,'use_residual':True}]
# ens_iterate_nas = [{}]

ensemble_iterate = meta_kw_fn(permute_fn(ens_iterate_hidden_sizes, ens_iterate_nas), 'model_params')
# ensemble_iterate = [{}]
# torch.optim.lr_scheduler.StepLR

kw_lib = torch_kw | optimizer_kw[0] | scheduler_kw[0] if LIB == 'torch' else skl_kw
print_dict = dict(print_less=True, epoch_print=True, print_only_mean=True)
# print_dict = dict(print_less=True, epoch_print=False, print_only_mean=True)
fitted_models = fit(epochs=epochs_train, **kw_lib, zero_lim=1e-4, iter_times=iter_train,
                    model_iter_options=model_iterate, ensemble_iter_options=ensemble_iterate,
                    batch_size=batch_size_train,
                    **ensemble_kw, **print_dict)

5
#
# dict_fit(epochs=200,
#          optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 1e-5},
#          model_cls=mm.LSTMModel, model_params={'hidden_size': 64, 'num_layers': 2},
#          zero_lim=1e-4)

# loss_fn = nn.L1Loss(),

# dict_fit(epochs=15, lr=1e-3, optimizer_fn=torch.optim.Adam, model_cls=mm.GRUModel)
# dict_fit(epochs=20,
#          optimizer_fn=torch.optim.Adam, optimizer_params={'lr': 1e-5},
#          model_cls=mm.LSTMModel, model_params={'hidden_size': 64, 'num_layers': 2},
#          zero_lim=1e-4)

print('END')
print('END')
# print(inputNames)
# print(targetNames)
# print(inputs.shape)
# [print('{}: {}'.format(e,p)) for e,p in zip(inputNames, inputs[0])]

# Some statistics
# cov = np.cov(inputs.transpose())
# mean = inputs.mean(axis=0)
# absMean = abs(inputs).mean(axis=0)
# span = inputs.max(axis=0)-inputs.min(axis=0)
# median = np.median(inputs)

# test_f_fn = lambda f: d_1m.fn(lambda x: tud.random_split(x.ds, [f, 1 - f])[0], update=False, level=4)
# dl_fn = lambda x: x.fn(tud.DataLoader, BATCH_SIZE, shuffle=False, update=False)
# test_short_agg_fn = lambda **kw: test_short.fn(tud.ConcatDataset, agg_level=1, update=False, **kw if not DEBUG else {})
