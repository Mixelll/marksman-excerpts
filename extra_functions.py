import copy
import io
import json
import pytz
import pickle as pickle_pkg
import numpy as np
import pandas as pd


from decimal import Decimal
from datetime import datetime, timedelta
from math import log10, floor, ceil
from pytz import timezone
from tzlocal import get_localzone
from collections.abc import Iterable

from inspect import signature
from collections import deque
from type_classes import *

# Useful
# wrapper class that adds a state to a function
class FuncWithState:
    def __init__(self, fn, state=None, state_kwarg=None):
        self.fn = fn
        self.state = state if state is not None else {}
        self.state_kwarg = state_kwarg

    def __call__(self, *args, **kwargs):
        if self.state_kwarg:
            return self.fn(*args, **kwargs, **{self.state_kwarg: self.state})
        # else: return self.fn(self.state, *args, **kwargs)


class FunctionArgBinder:
    """ lambda and nested functions replacement for pickling
    Binds args at init to fn(*args), also passes kwargs to fn.
    Also binds args to kwargs, and kwargs to args, and changes call order."""
    def __init__(self, function, *args, args2kwargs=(), kwargs2args=None, arg_call_first=True, arg_bound_first=True,
                 arg_changer_fn=None, arg_changer_args_in=None, arg_changer_args_out=None,
                 **kwargs):
        self.function = function
        self.arg_binding = args2kwargs if args2kwargs else ()
        self.kwarg_binding = kwargs2args if kwargs2args else {}
        self.arg_call_first = arg_call_first
        self.arg_bound_first = arg_bound_first
        self.args = args
        self.kwargs = kwargs
        self.arg_changer_fn = arg_changer_fn
        self.arg_changer_fn_in = arg_changer_args_in
        self.arg_changer_fn_out = arg_changer_args_out if arg_changer_args_out else arg_changer_args_in
    def __call__(self, *args, **kwargs):
        bound_args = []
        if self.kwarg_binding:
            bind_keys = list(self.kwarg_binding.keys())
            for k in bind_keys:
                if k in kwargs:
                    bound_args.append(kwargs[k])
                    del kwargs[k]
                else:
                    bound_args.append(self.kwarg_binding[k])

        bound_kwargs = {}
        if self.arg_binding:

            for i, arg_name in enumerate(args):
                bound_kwargs[self.arg_binding[i]] = arg_name
            if len(args) > len(self.arg_binding):
                args = args[len(self.arg_binding):]
            else:
                args = []

        total_kwargs = {} | self.kwargs | kwargs | bound_kwargs

        if self.arg_bound_first:
            args_new = bound_args + list(args)
        else:
            args_new = list(args) + bound_args

        if self.arg_call_first:
            args_final = args_new + list(self.args)
        else:
            args_final = list(self.args) + args_new

        if self.arg_changer_fn:
            _args_in = []

            if self.arg_changer_args_in:
                for arg_name in self.arg_changer_args_in:
                    if isinstance(arg_name, str):
                        _args_in.append(total_kwargs[arg_name])
                    elif not isinstance(arg_name, int):
                        _args_in.append(args_final[arg_name])
            _args_out = self.arg_changer_fn_in(*_args_in)

            if _args_out is not None and arg_changer_args_out:
                for i, arg_val in enumerate(self._args_out):
                    arg_name = self.arg_changer_args_out[i]
                    if isinstance(arg_val, str):
                        total_kwargs[arg_name] = arg_val
                    else:
                        args_final[arg_name] = arg_val

        return self.function(*args_final, **total_kwargs)

# lambda replacement for pickling
class AccessorK:
    def __init__(self, key):
        self.key = key
    def __call__(self, x):
        return x[self.key]

# lambda replacement for pickling
class InRangeFn:
    def __init__(self, low=None, high=None):
        self.low = low
        self.high = high

    def __call__(self, x):
        if self.low is not None and self.high is not None:
            return self.low < x < self.high
        elif self.low is not None:
            return x < self.low
        elif self.high is not None:
            return x > self.high
        else:
            return True


def str_ex_num(x):
    return x if isinstance(x, (str, Number)) else str(x)

def pickle(obj, file=None):
    def _pickle(f, *args, **kwargs):
        with f(*args, **kwargs) as _file:
            pickle_pkg.dump(obj, _file)
            _file.seek(0)
            return file

    if file is None:
        return _pickle(io.BytesIO)
    else:
        return _pickle(open, file, 'wb')

def pickle_load(file):
    return pickle_pkg.load(open(file, 'rb'))

# Useful
class SortedDeque(deque):
    def __init__(self, iterable=(), maxlen=None, compare=lambda x, y: x > y, accessor=lambda x: x, assert_fn=None):
        super().__init__(iterable, maxlen=maxlen)
        self.compare = compare
        self.accessor = accessor
        self.assert_fn = assert_fn

    def insert_sorted(self, new_element, check_only=False):
        accessed_element = self.accessor(new_element)

        if self.assert_fn is not None:
            if not self.assert_fn(accessed_element):
                return False

        # Find the correct position for the new element and check or insert it
        for i, element in reversed(list(enumerate(self))):
            if self.compare(accessed_element, self.accessor(element)):
                if check_only:
                    return True  # The new element would be inserted
                # Ensure there's space to insert the new element by removing from the start if at maxlen
                if self.maxlen is not None and len(self) == self.maxlen:
                    self.popleft()  # Remove one item from the start to make space
                self.insert(i, new_element)
                return None

        if self.maxlen is None or len(self) < self.maxlen:
            if check_only:
                return True
            else:
                self.insert(0, new_element)
                return None
        else:
            return False if check_only else None



def get_sub_lists_i(lst, i=0):
    return list(list(zip(*lst))[i])

# Useful
class StdoutRedirect:
    def __init__(self):
        self.virtual_file = io.StringIO()
        self.original_stdout = sys.stdout

    def start_redirect(self):
        sys.stdout = self

    def write(self, data):
        self.virtual_file.write(data)
        self.original_stdout.write(data)

    def flush(self):
        self.virtual_file.flush()
        self.original_stdout.flush()

    def get_contents(self):
        return self.virtual_file.getvalue()

    def close(self):
        self.virtual_file.close()

    def reset_stdout(self):
        sys.stdout = self.original_stdout

# Useful
class StdoutRedirectWithExit(StdoutRedirect):

    def __enter__(self):
        sys.stdout = self  # Redirect sys.stdout to this object
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.reset_stdout()  # Reset sys.stdout to its original state



class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()  # Ensures the data is written and visible immediately

    def flush(self):
        for stream in self.streams:
            stream.flush()

def update_dict(dict1, dict2):
    for k in dict1:
        if k in dict2:
            dict1[k] = dict2[k]
    return dict1


def kw_wrapper(fn, **kw):
    return lambda *ar, **kw2: fn(*ar, **kw, **kw2)


def max_nesting_level(lst):
    def get_depth(lst, current_depth):
        max_depth = current_depth
        for element in lst:
            if isinstance(element, list):
                depth = get_depth(element, current_depth + 1)
                max_depth = max(max_depth, depth)
        return max_depth

    return get_depth(lst, 1)

def is_slice_in_list(s, lis):
    len_s = len(s)
    return any(s == lis[i:len_s + i] for i in range(len(lis) - len_s + 1))


def is_list_in_list_order(a, b):
    len_a = len(a)
    for i in range(len(b) - len_a + 1):
        if a == b[i:i + len_a]:
            return True
    return False


def unique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

# Useful
def flatten(xs, type_=Iterable, return_outer=False, temp_list=None, return_list=True):
    if return_outer:
        temp_list = []
        return flatten(xs, type_=type_, return_outer=False, temp_list=temp_list, return_list=True), temp_list[0]
    elif return_list:
        return list(flatten(xs, type_=type_, return_outer=False, temp_list=temp_list, return_list=False))
    else:
        return _flatten(xs, type_=type_, temp_list=temp_list)

def _flatten(xs, type_=Iterable, temp_list=None):
    for x in xs:
        if isinstance(x, type_) and not isinstance(x, (str, bytes)):
            yield from _flatten(x, type_=type_, temp_list=temp_list)
        else:
            if temp_list is not None and len(temp_list) == 0:
                temp_list.append(xs)
            yield x

def sequential_permutations(vector):
    return [tuple(vector[:i+1]) for i in range(len(vector))]


def get_exponent(number):
    sign, digits, exponent = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def get_mantissa(number):
    return Decimal(number).scaleb(-get_exponent(number)).normalize()


# Useful
# decorator to copy the signature of a function to another function
def copy_signature(source_fct):
    def _copy(target_fct):
        target_fct.__signature__ = signature(source_fct)
        return target_fct
    return _copy


def join_dataframes_on_index(dataframe_dict, single=lambda x: 'date' in x):

    dataframe_dict.fn_k_v(lambda k, v: add_suffix_to_column_names(v, prefix=k, ignore=single))
    dataframes = list(dataframe_dict.values())
    result_df = dataframes[0]
    for df in dataframes[1:]:
        c = df.columns
        result_df = pd.merge(result_df, df.drop(columns=c[c.map(single)]), how='inner', left_index=True, right_index=True)

    return result_df

def add_suffix_to_column_names(dataframe, prefix=None, suffix=None, sep='_', ignore=None):
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame")

    prefix_str = '' if prefix is None else prefix + sep
    suffix_str = '' if suffix is None else sep + suffix
    if callable(ignore):
        new_columns = [prefix_str + col + suffix_str
                       if not ignore(col) else col
                       for col in dataframe.columns]
    else:
        new_columns = [prefix_str + col + suffix_str
                       if ignore is None or col not in ignore else col
                       for col in dataframe.columns]
    dataframe.columns = new_columns
    return dataframe

def to_ny_tz(ts, local=True):
    return to_timezone(ts, 'America/New_York', local=local)

def ny_localize(ts):
    return pytz.timezone('America/New_York').localize(ts)

def now_as_tz(ts):
    return to_timezone(datetime.now(), get_timezone(ts), local=True)



def df_freq_dict(df, index = 'date'):
    freqs = ['20Y','10Y', '5Y', '3Y', 'Y', 'Q', 'M', 'W', '2D', 'D', '3H', 'H',
            '15m', '5m', '1m', '15s', '5s', '1s']
    deltas = str2td(freqs)
    dfInd = df_return_ind_col(df, index)
    splitter = df_create_freq_splitter(df, index)

    a = [abs(max(dfInd)-min(dfInd) - x) for x in deltas]
    ind1 = a.index(min(a))
    a = [abs(dfInd.to_series().diff().value_counts().index[0] - x) for x in deltas]
    ind2 = a.index(min(a))
    split = []
    stat_dict_inst = {}
    # print(datetime.now())
    for i in range(ind1+1,ind2-1):
        # print(freqs[i])
        # print(deltas[ind1]/deltas[i])
        if deltas[ind1]/deltas[i] <= 5000:
            split = splitter(freqs[i])
        if split:
            stat_dict_inst = dict_merge(stat_dict_inst, stat_dict(clean_split(split, index),
                                    index, arrayOp=lambda x: {freqs[i]: x}))

    return stat_dict_inst

def clean_split(split, index=None, num = 0.6):
    # print(split)
    split_clean = [x for x in split if len(x)]

    split_len = [len(x) for x in split_clean]
    split_cleanest = []
    keep = None
    le_mean = sum(split_len) / len(split_len)
    if num < 1:
        num *= le_mean

    for x in split_clean:
        if keep is not None:
            if len(split_cleanest):
                dfIndK = df_return_ind_col(keep, index)
                dfIndX = df_return_ind_col(x, index)
                dfIndPrev = df_return_ind_col(split_clean[-1], index)
                distanceXK = min(dfIndX) - max(dfIndK)
                distanceKP = min(dfIndK) - max(dfIndPrev)
                if distanceXK < distanceKP:
                    x = pd.concat([keep, x])
                else:
                    split_cleanest[-1] = pd.concat([split_cleanest[-1], keep])
            else:
                x = pd.concat([keep, x])

        if len(x) < num:
            keep = x
        else:
            split_cleanest.append(x)
            keep = None

    # print([len(x) for x in split])
    # print([len(x) for x in split_cleanest])
    return split_cleanest

def df_create_freq_splitter(df, index=None):
    if index is None:
        index = df.index.names[0]
    if df_check_ind(df, index):
        return lambda f: [i for _,i in df.groupby(pd.Grouper(level=index, freq=f))]
    else:
        return lambda f: [i for _,i in df.groupby(pd.Grouper(key=index, freq=f))]


def stat_dict(dfArray, index=None, arrayOp = None):
    if arrayOp:
        def intervals(op): return arrayOp(df_interval_split_op(dfArray, op, index=index))
    else:
        def intervals(op): return df_interval_split_op(dfArray, op, index=index)

    return {
        # 'pd.cov':  intervals(lambda x: x.cov()),
        'pd.describe': intervals(lambda x: x.describe())
    }

def replace_inf_nan(df, what=None):
    return df.replace([np.inf, -np.inf, np.nan], what)

def df_interval_split_op(dfArray, op, index=None):
    out = {}
    for df in dfArray:
        if len(df):
            dfInd = df_return_ind_col(df, index)
            out[pd.Interval(min(dfInd), max(dfInd))] = replace_inf_nan(op(df))
    return out


def df_return_ind_col(df, index=None):
    if df_check_ind(df, index):
        return df.index
    else:
        return df[index]

def df_check_ind(df, index):
    if index in df.index.names:
        return True
    else:
        try:
            df[index]
            return False
        except:
            raise ValueError(f'Name {index} not found in indices or column names')


def df_index_wrapper(df, index=None, to_series=False):
    if isinstance(df,  pd.Index):
        if to_series:
            return df.to_series()
        else:
            return df

    if index is None:
        if to_series:
            return df.index.to_series()
        else:
            return df.index
    elif isinstance(index, str):
        return df[index]


# def df_index_wrapper(df, func, index=None):
#
#     if index is None:
#         return func(df.index.to_series()).index
#
#     elif isinstance(index, str):
#         return func(df[index]).index


def df_time_mask(df, t, return_ind=True, **kwargs):
    ind = df_index_wrapper(df, **kwargs)
    contained = t.pd_contained(ind)
    if return_ind and not isinstance(contained, pd.Index):
        return ind[contained]
    return contained

def df_freq_split(df, freq, return_ind=False, **kwargs):
    ind = df_index_wrapper(df, to_series=True, **kwargs) if return_ind else df
    ind_fn = lambda x: x.index
    dfx_fn = ind_fn if return_ind else lambda x: x
    return {pd.Interval(left=min(ind_fn(x)),  right=max(ind_fn(x)), closed='both'): dfx_fn(x)
            for _, x in ind.groupby(pd.Grouper(freq=freq)) if not x.empty}

def pd_merge_intervals(intervals):
    mn = min(map(lambda x: x.left, intervals))
    mx = max(map(lambda x: x.right, intervals))
    return pd.Interval(mn,  mx, closed='both')

def iter2list(o):
    # if (isinstance(o, range) or isinstance(o, types.GeneratorType)) and not isinstance(o, str):
    if not isinstance(o, str):
        return list(iter(o))
    return o


def str2pd_interval(o, tz='America/New_York'):
    if o[0] not in '([' or  o[-1] not in '])':
        raise ValueError(f'Object {str(o)} of {str(type(o))} does not represent \
                        a pd.Interval')
    oo = o.split(',')
    def f(x): return to_timezone(pd.Timestamp(x), tz)
    return pd.Interval(f(oo[0][1:]), f(oo[1][:-1]))

    # return o

def pd_interval2str(o):
    if isinstance(o, pd.Interval):
        oo = str(o)
        return oo[0] + str(o.left) + ',' + str(o.right) + oo[-1]
    return o

def element_array_merge(a, b):
    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]

    a.extend(b)
    return a

def dict_merge(a, b, m=0):
    "merges b into a"
    for key in b:
        if key in a:
            aD, bD = isinstance(a[key], dict), isinstance(b[key], dict)
            if a[key] is None:
                a[key] = b[key]
            elif aD and bD:
                dict_merge(a[key], b[key])
            elif aD or bD:
                if m // 10 == 0:
                    if m % 10 == 0:
                        a[key] = b[key]
                    elif m % 10 == 1 and a[key] != b[key]:
                            a[key] = element_array_merge(a[key], b[key])
                    elif m % 10 == 2:
                            a[key] = element_array_merge(a[key], b[key])
                elif m // 10 == 1 and not aD and bD:
                        a[key] = b[key]
            else:
                if m % 10 == 0:
                    a[key] = b[key]
                elif m % 10 == 1 and a[key] != b[key]:
                        a[key] = element_array_merge(a[key], b[key])
                elif m % 10 == 2:
                        a[key] = element_array_merge(a[key], b[key])
        else:
            a[key] = b[key]
    return a

def object_captain_hook(o, default = [(str2pd_interval, pd.DataFrame.from_dict),
                        str2pd_interval]):
    if callable(default):
        default = [default]

    for d in default:
        if callable(d) or len(d)==1:
            if not callable(d):
                d = d[0]
            od = {}
            for k,v in o.items():
                k0 = k
                try:
                    k = d(k)
                except:
                    pass
                try:
                    v = d(v)
                except:
                    pass
                try:
                    od[k] = v
                except:
                    od[k0] = v
        else:
            od = {}
            for k,v in o.items():
                try:
                    kd =  d[0](k)
                    if kd:
                        od[kd] = d[1](v)
                    else:
                        od[k] = v
                except:
                    od[k] = v
        o = od
    return od

def iter_length(*args):
    out = []
    for x in args:
        if x is None:
            out.append(0)
        elif isinstance(x, str):
            out.append(1)
        else:
            out.append(len(list(copy.deepcopy(x))))
    return out


def multiply_iter(v, n=1):
    if isinstance(v, str): return [v]*n
    o = copy.deepcopy(v)

    try:
        iter(v)
        return o
    except TypeError:
        return [o]*n


def mbool(val, st=' ', useBool=True):
    if isinstance(val, str):
        return bool(val.strip(st))
    elif useBool:
        return bool(val)
    return True


def n_times(list, f):
    out = []
    for e in list:
        out.extend([e] * f)
    return out


def round_to(x, sig):
    if x == 0:
        return 0
    try:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    except TypeError:
        if isinstance(x, np.ndarray):
            return np.array(list(map(lambda y: round_to(y, sig), x)))
        else:
            return x.__class__(map(lambda y: round_to(y, sig), x))


def get_timezone(x):
    typeStr = str(type(x)).lower()
    if isinstance(x, str):
        out = x
    elif typeStr.find('pandas') != -1:
        try:
            out = x.dt.tz
        except:
            out = x.tz
    elif typeStr.find('tzfile') != -1:
        out = x.zone
    else:
        out = x.tzinfo.zone
    return out


def to_timezone(inp, tz=None, naive=False, local=False):
    type_str = str(type(inp)).lower()
    localize_tz = str(get_localzone()) if local else 'UTC'
    if tz is None:
        tz = localize_tz
    if 'pandas' in type_str:
        try:
            if inp.tz is None:
                out = inp.tz_localize(localize_tz,
                                      nonexistent='shift_backward').tz_convert(str(tz))
            else:
                out = inp.tz_convert(tz)
            if naive:
                out = out.tz_localize(None)
        except:
            if inp.dt.tz is None:
                out = inp.dt.tz_localize(localize_tz, ambiguous='infer',
                                         nonexistent='shift_backward').dt.tz_convert(str(tz))
            else:
                out = inp.dt.tz_convert(tz)
            if naive:
                out = out.dt.tz_localize(None)

    else:
        if inp.tzinfo is None:
            out = timezone(localize_tz).localize(inp)
        else:
            out = inp
        if isinstance(tz, str):
            out = out.astimezone(timezone(tz))
        else:
            out = out.astimezone(tz)
        if naive:
            out = out.replace(tzinfo=None)
    return out


def td2str_tq(time_quants, time_quants_vals, delta, unit=None, space=False, no_prefix=False, by_ratio=False, r=1):
    time_quants = [x for _, x in sorted(zip(time_quants_vals, time_quants),
                                        key=lambda pair: pair[0], reverse=True)]
    time_quants_vals = sorted(time_quants_vals, reverse=True)
    fInd = time_quants.index(unit) if unit else -1
    deltas = delta if isinstance(delta, (list, tuple)) else [delta]
    out = []
    ind = 0 if by_ratio else fInd
    for d in deltas:
        if any(char.isdigit() for char in time_quants[0]):
            a = [abs(x-d) for x in time_quants_vals]
            out.append(time_quants[a.index(min(a))])
        else:
            for i in range(0, len(time_quants)):
                if by_ratio:
                    if (time_quants_vals[min(0, i-1)] / d) >= (d / time_quants_vals[i]):
                        ind = max(fInd,i)
                        continue
                else:
                    if (d / time_quants_vals[i]) >= r:
                        ind = max(fInd,i)
                        break
            if no_prefix:
                prefix = ''
            else:
                prefix = str(ceil(d / time_quants_vals[ind]))
                if space:
                    prefix += ' '
            out.append(prefix + time_quants[ind])

    if isinstance(delta, (list, tuple)):
        return out
    else:
        return out[0]


def str2td(str_list):
    td = timedelta
    if isinstance(str_list, td):
        return str_list
    forms = [(('Y','y','year'), td(days=365)), (('Q','q','quarter'), td(days=90)),
            (('M','month'), td(days=30)), (('W','w','week'), td(weeks=1)),
            (('D','d','day'), td(days=1)), (('H','h','hour'), td(hours=1)),
            (('T','m','min','minute'), td(minutes=1)),
            (('S','s','sec','second'), td(seconds=1))]
    def fi(s): return [dt for st,dt in forms if s in st][0]
    # dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    out = []
    strs = [str_list] if isinstance(str_list, str) else str_list
    for s in strs:
        if isinstance(s, td):
            out.append(s)
            continue
        if len(s.lstrip('0123456789 .')) > 1:
            s = s.rstrip('s')
        try:
            if any(c.isdigit() for c in s):
                ss = s.lstrip('0123456789 .')
                out.append(float(s.replace(ss,'')) * fi(ss))
            else:
                out.append(fi(s))
        except:
            err = f'Name {s.lstrip("0123456789 .")} not found in time delta names'
            raise ValueError(err) from None
    if isinstance(str_list, str):
        return out[0]
    return out


def ib_hist_duration(delta, **kwargs):
    if isinstance(delta, str):
        return delta
    time_quants = ['Y', 'M', 'W', 'D', 'S']
    return td2str_tq(time_quants, str2td(time_quants), delta, space=True, **kwargs)


# def td_str(delta, **kwargs):
#     if isinstance(delta, str):
#         return delta
#     time_quants = ['Y', 'M', 'W', 'D', 'H', 'm', 's']
#     return td2str_tq(time_quants, str2td(time_quants), delta, **kwargs)

def td_str(delta, time_quants = ('Y', 'M', 'W', 'D', 'H', 'm', 's'), **kwargs):

    if isinstance(delta, str) or isinstance(delta, (list, tuple)) and isinstance(delta[0], str):
        delta = str2td(delta)
    return td2str_tq(time_quants, str2td(time_quants), delta, **kwargs)

def bars_size(delta, **kwargs):
    time_quants = ['1 secs', '5 secs', '10 secs', '15 secs', '30 secs', '1 min',
                    '2 mins', '3 mins', '5 mins', '10 mins', '15 mins', '20 mins',
                    '30 mins', '1 hour', '2 hours', '3 hours', '4 hours',
                    '8 hours','1 day', '1 week', '1 month']
    if isinstance(delta, str) or isinstance(delta, (list, tuple)) and isinstance(delta[0], str):
        delta = str2td(delta)
    return td2str_tq(time_quants, str2td(time_quants), delta, **kwargs)
