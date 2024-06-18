import json
from datetime import date, datetime, time, timedelta
from typing import NamedTuple

import pandas as pd
import pytz
import talib

import extra_functions as me
import m_objects as mo
from extra_functions import ny_localize
from fin_dicts import TimeSeriesDict
from m_objects import TimeInterval


class USTimes:
    tz = 'America/New_York'
    market_hours = TimeInterval([time(hour=9, minute=30), time(hour=16, minute=0)], tz=tz)
    orth_hours = TimeInterval([time(hour=9, minute=30), time(hour=16, minute=0)], tz=tz)
    open35 = TimeInterval([time(hour=9, minute=30), time(hour=10, minute=5)], tz=tz)
    open_noon = TimeInterval([time(hour=9, minute=30), time(hour=12, minute=30)], tz=tz)
    noon = TimeInterval([time(hour=12, minute=0), time(hour=14, minute=0)], tz=tz)
    close_hour = TimeInterval([time(hour=15, minute=0), time(hour=16, minute=0)], tz=tz)
    close35 = TimeInterval([time(hour=15, minute=25), time(hour=16, minute=0)], tz=tz)


    extended_hours = TimeInterval([time(hour=8, minute=00), time(hour=18, minute=0)], tz=tz)
    extended_hours_long = TimeInterval([time(hour=7, minute=00), time(hour=19, minute=0)], tz=tz)
    pre_mkt35 = TimeInterval([time(hour=8, minute=55), time(hour=9, minute=30)], require_oh=True, tz=tz)
    pre_mkt60 = TimeInterval([time(hour=8, minute=30), time(hour=9, minute=30)], require_oh=True, tz=tz)


def dict_multi_time_mask(d, times, index=None):
    df = d.df
    fn = me.df_time_mask
    # df_index = fn(df, t, index=index)
    d.update({t: TimeSeriesDict(df = df, t=t, index_fn=lambda o: fn(o.df, o.t, index=index)) for t in times})


def dict_freq_split(d, frequencies, index=None, save_split_df=False, min_freq_ratio=0.5, append_small_splits=False):
    df = d.df
    t_cls = d.t_cls
    lambda_x = lambda x: x

    def append_small_splits_fn(_df_split, input_fn=lambda_x, output_fn=lambda_x, ratio_fn=None, ratio_input_fn=lambda_x):
        if ratio_fn is None:
            return
        prev_k = None
        prev_append = None
        new_df_split = {}
        for i, (k, v) in enumerate(_df_split.items()):
            prev_k_fixed = prev_k
            k_new = k
            v_new = v
            if prev_append is not None:
                if prev_k_fixed in new_df_split:
                    del new_df_split[prev_k_fixed]
                curr = input_fn(v_new)
                k_new = me.pd_merge_intervals([prev_k, k_new])
                v_new = output_fn(pd.concat([input_fn(prev_append), curr]))

            if ratio_fn(ratio_input_fn(v_new)):
                curr = input_fn(v_new)
                if prev_k is not None and prev_append is None:
                    if prev_k in new_df_split:
                        prev = new_df_split[prev_k]
                        del new_df_split[prev_k]
                    else:
                        prev = input_fn(_df_split[prev_k])
                    k_new = me.pd_merge_intervals([prev_k, k_new])
                    v_new = output_fn(pd.concat([prev, curr]))
                    if ratio_fn(ratio_input_fn(v_new)):
                        prev_k = k_new
                        prev_append = v_new
                    else:
                        prev_k = k_new
                        prev_append = None
                        new_df_split[k_new] = v_new
                else:
                    prev_k = k_new
                    prev_append = v_new
            else:
                prev_k = k_new
                prev_append = None
                new_df_split[k_new] = v_new

        if prev_append is not None:
            new_df_split[prev_k] = prev_append
        _df_split.clear()
        _df_split.update(new_df_split)

    for f in frequencies:
        ratio_fn_bool = lambda x: (max(x) - min(x)) / me.str2td(f) <= min_freq_ratio
        lambda_tsd = lambda *a, **kw: TimeSeriesDict(*a, **kw, t_cls=t_cls)
        d_freq = lambda_tsd(df = df, df_index=d.df_index)
        if append_small_splits:
            if save_split_df:
                df_split = me.df_freq_split(df.loc[d.df_index], f, index=index)
                append_small_splits_fn(df_split, ratio_fn=ratio_fn_bool, ratio_input_fn=lambda x: x.index)
                # d_freq[k] = TimeSeriesDict(df=v, df_index=v.index)
                d_freq.update({k: lambda_tsd(df = v, df_index=v.index) for k, v in df_split.items()})
            else:
                df_split = me.df_freq_split(d.df_index, f, return_ind=True, index=index)
                append_small_splits_fn(df_split, lambda x: x.to_series(), lambda x: x.index, ratio_fn_bool)
                d_freq.update({k: lambda_tsd(df = df, df_index=v) for k, v in df_split.items()})
        else:
            if save_split_df:
                d_freq.update({k: lambda_tsd(df = v, df_index=v.index)
                               for k, v in me.df_freq_split(df.loc[d.df_index], f, index=index).items()})
            else:
                d_freq.update({k: lambda_tsd(df = df, df_index=v)
                               for k, v in me.df_freq_split(d.df_index, f, return_ind=True, index=index).items()})
        d[f] = d_freq


def add_sma_columns(nes_d, columns, settings=(10, 30, 50, 125, 200)):
    day_sma = settings
    name_list2 = [(f'{c}_{sm}D', c, sm) for c in columns for sm in day_sma]
    name_list = [x[0] for x in name_list2]
    def fn(df, nesting, /):
        for n, c, sm in name_list2:
            df[n] = nes_d['1m'].get_n(nesting)[c].rolling(sm).mean()
    nes_d['1D'].fn(fn, update=False)

    return name_list


def add_ta_columns(nes_d):
    once_flag_container = [True]
    out = []
    def fn(df):
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
        df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2,nbdevdn=2, matype=0)
        if once_flag_container[0]:
            once_flag_container[0] = False
            out[0] = [c for c in df.columns if c not in columns0]


    nes_d['1D'].fn(fn, update=False)

    return out[:]


freqs = ['20Y','10Y', '5Y', '3Y', 'Y', 'Q', 'M', 'W', '2D', 'D', '3H', 'H',
        '15m', '5m', '1m', '15s', '5s', '1s']


# df generate_data_dict()


class Time(mo.DatetimeInterval):
    def in_transform(self, other):
        return other.time()

    def series_in_transform(self, s):
        return s.dt.time

