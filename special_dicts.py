import itertools as it
import warnings
from copy import copy, deepcopy
from collections import defaultdict
from inspect import signature
from operator import itemgetter
# from typing import List

import extra_functions as ef

# from type_classes import *
from numbers import Number
from collections.abc import Iterable


class NestedDict(dict):
    origin = None
    parent = None
    nesting = None
    name = None

    def iter(self):
        return self.items()

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.set_links()

    def set_name(self, name):
        self.name = name
        return self

    def set_links(self):
        kw = dict(update=False, to_level=True, on_leaf=False)
        self.fn(lambda x,: setattr(x, 'origin', self), **kw)
        self.fn(lambda x, nes, parent, /: setattr(x, 'parent', parent), **kw)
        self.fn(lambda x, nes, /: setattr(x, 'nesting', nes), **kw)
        return self

    def update_new(self, update_dict, level=0):
        flat_self = self.flatten()
        flat_update = update_dict.flatten()
        def update_fn(x):
            for k, v in flat_update.items():
                if k not in flat_self:
                    x.set_n(k, v)
        return self.fn(update_fn, level=level)

    def ikey(self, level=0, keys=0):
        if level==0:
            return self.fkey(keys=keys)
        else:
            return self.fval(keys=0).ikey(level=level-1, keys=keys)

    def ival(self, level=0, keys=0):
        if level==0:
            return self.fval(keys=keys)
        else:
            return self.fval(keys=0).ival(level=level-1, keys=keys)
    def fkey(self, keys=0):
        self_keys = list(self.keys())
        if isinstance(keys, list):
            return [self_keys[i] for i in keys]
        else:
            return self_keys[keys]

    def fval(self, keys=0):
        self_keys = list(self.keys())
        if isinstance(keys, list):
            return [self[self_keys[i]] for i in keys]
        else:
            return self[self_keys[keys]]

    def get_n(self, keys, default=None):
        if isinstance(keys, (str, Number)):
            return self.get(keys, default)
        x = self
        for k in keys:
            if isinstance(x, NestedDict):
                x = x.get(k, default)
            else:
                return default
        return x

    def get_n_partial(self, keys):
        for i in range(len(keys)):
            x = self.get_n(keys[i:])
            if x is not None:
                return x

    def set_n(self, keys: list, set_val, objects=None):
        x = self
        for i, k in enumerate(keys[:-1]):
            if isinstance(x, NestedDict):
                if k in x.keys():
                    x = x[k]
                elif objects is not None and isinstance(objects[i], NestedDict):
                    x[k] = objects[i].copy(deep=False)
                    x[k].clear()
                    x = x[k]
                else:
                    x[k] = self.__class__()
                    x = x[k]
            else:
                raise KeyError
        x[keys[-1]] = set_val
        return self

    def del_n(self, keys: list):
        if isinstance(keys, (str, Number)):
            del self[keys]
            return self
        x = self
        for k in keys[:-1]:
            if isinstance(x, NestedDict):
                x = x[k]
            else:
                raise KeyError
        del x[keys[-1]]
        return self


    def fn_v(self, fn):
        for k in self.keys():
            self[k] = fn(self[k])
        return self

    def fn_k(self, fn):
        for k in self.keys():
            self[fn(k)] = self.pop(k)
        return self

    def fn_k_v(self, fn):
        for k in self.keys():
            self[k] = fn(k, self[k])
        return self


    def fn(self, func, *args, replace_keys=True, update=True, update_none=False, on_leaf=None,
           ignore_keys=None, keep_ignored=False, discard_none=False,
           func_n_args=None, on_empty_dict=False, nesting=(), cast_func=None,
           no_return=False,  cast_nested=False, ignore_none=True,
           level=None, to_level=False, parent=None,
           agg_level=None, agg_fn=None, agg_chain_iterable=True,
           fn_args=None, fn_kwargs=None, fn_sig=None,  **kwargs):
        """
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        WHEN ADDING KEYWORD ARGUMENTS, ADD THEM TO KW BELOW AS WELL
        except: nesting, parent, level
        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        """
        if level is None:
            level = 9999
        if agg_level is None:
            agg_level = 9999
        if agg_level < 0:
            agg_level = self.count_levels() + agg_level
            if agg_level < 0:
                warnings.warn('abs(agg_level) is more than the number of levels in the dict, agg_level set to 0')
                agg_level = 0
        elif agg_level > 0:
            agg_level = agg_level - 1
        if on_leaf is None and agg_level > 1000:
            on_leaf = True
            on_leaf_pass = None
        else:
            on_leaf_pass = on_leaf

        if fn_args is None:
            fn_args = tuple()
        if fn_kwargs is None:
            fn_kwargs = {}
        if cast_nested:
            cast_func = self.__class__.from_dict

        if no_return:
            update = False
            cast_func = dict


        inst_method = isinstance(func, str)

        if fn_sig is None:
            func4sig = getattr(self, func) if inst_method else func
            try:
                pos_param = sum(1 for p in signature(func4sig).parameters.values() if p.kind == p.POSITIONAL_ONLY)
                pos_or_key_param = sum(1 for p in signature(func4sig).parameters.values() if p.kind in {
                    p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL})
                if inst_method:
                    sig = max(pos_param -1, 0)
                else:
                    sig = pos_param or pos_or_key_param
            except ValueError:
                pos_param = 0
                if inst_method:
                    sig = 0
                else:
                    sig = 1
            if func_n_args is not None:
                sig = func_n_args

            fn_sig = sig, pos_param
        else:
            sig, pos_param = fn_sig

        kw = dict(replace_keys=replace_keys, update=update, update_none=update_none, on_leaf = on_leaf_pass,
        ignore_keys=ignore_keys, keep_ignored=keep_ignored, discard_none=discard_none,
        func_n_args=func_n_args, on_empty_dict=on_empty_dict, cast_func=cast_func,
        ignore_none=ignore_none,
        to_level=to_level,
        agg_level=agg_level, agg_fn=agg_fn, agg_chain_iterable=agg_chain_iterable,
        fn_args=fn_args, fn_kwargs=fn_kwargs, fn_sig=fn_sig, **kwargs)


        if inst_method:
            func_str = func
            if sig == 0:
                _func = lambda x, *_args: getattr(x, func_str)(*args, *fn_args, **fn_kwargs, **kwargs) if isinstance(x, NestedDict) else x
            if sig >= 1:
                _func = lambda x, *_args: getattr(x, func_str)(*_args, *args, *fn_args, **fn_kwargs, **kwargs) if isinstance(x, NestedDict) else x
        else:
            if sig == 0:
                _func = lambda x, *_args: func(*args, *fn_args, **fn_kwargs, **kwargs)
            elif pos_param == 2:
                _func = lambda x, *_args: func(x, _args[0], *args, *fn_args, **fn_kwargs, **kwargs)
            elif pos_param >= 3:
                _func = lambda x, *_args: func(x, *_args, *args, *fn_args, **fn_kwargs, **kwargs)
            elif sig >= 1:
                _func = lambda x, *_args: func(x, *args, *fn_args, **fn_kwargs, **kwargs)


        empty_flag = on_empty_dict and len(self) == 0
        out = _func(self, nesting, parent) if empty_flag or to_level or level == 0 else self
        out = out if out is not None else self
        if out == 'secret_break_keyword777':
            return None
        if out is None:
            return None
        if level != 0 and not empty_flag:
            if level > 0:
                level = max(0, level - 1)
            out_temp = {}
            for k, v in out.items():
                if ignore_keys is not None and k in ignore_keys:
                    if keep_ignored:
                        out_temp[k] = v
                    continue
                apply_fn_flag = False
                if isinstance(v, NestedDict) and level < 0:
                    if v.count_levels() == -level:
                        apply_fn_flag = True
                if not apply_fn_flag and isinstance(v, NestedDict) and (level or to_level):
                    out_temp[k] = v.fn(func, *args, nesting=nesting + (k,), parent=self, level=level, **kw)
                else:
                    if apply_fn_flag or (isinstance(v, NestedDict) or on_leaf) and not to_level and not (ignore_none and v is None):
                        temp = _func(v, nesting + (k,), self)
                        out_temp[k] = temp if not update or update and (update_none or temp is not None) else v
                    else:
                        out_temp[k] = v
                    if discard_none and out_temp[k] is None:
                        del out_temp[k]

            if out is self:
                out = out_temp
            else:
                out.clear()
                out.update(out_temp)
        if agg_level == 0 and not empty_flag:
            out_vals = out.values()
            if agg_chain_iterable and all(map(lambda x : isinstance(x, Iterable), out_vals)):
                out_vals = it.chain(*out_vals)
            if agg_fn is None:
                return _func(list(out_vals), nesting)
            else:
                if callable(agg_fn) and not agg_fn == list :
                    return agg_fn(list(out_vals))
                else:
                    return list(out_vals)


        if out is self:
            return self
        if cast_func is None:
            if isinstance(out, dict):
                out = self.__class__(out)
        else:
            out = cast_func(out)
        if update:
            # if update_none or out is not None:
            if replace_keys:
                self.clear()
            self.update(out)
            return self
        else:
            return out


    def fn_other(self, other, func=lambda x, y: x - y, **kwargs):
        return self.fn(lambda x, nesting, /: func(x, other.get_n(nesting)), **kwargs)

    def tree_index(self, index):
        if isinstance(index, NestedDict):
            flat_index = index.flatten()
            out = self.__class__()
            for k, v in flat_index.items():
                if v:
                    out.set_n(k, self.get_n(k))
        return out


    @classmethod
    def from_dict(cls, nested_dict, levels=9999, stop_on_cls=False):
        levels = levels - 1
        condition = lambda x: isinstance(x, dict) and (not stop_on_cls or not isinstance(x, NestedDict))
        return cls({k: cls.from_dict(v, levels) if condition(v) and levels else v for k, v in nested_dict.items()})

    @classmethod
    def from_nested(cls, x, nesting=(), k_func=None, def_val=''):
        if isinstance(x, list) and x:
            x = x[0]

        def create_instance_k_func():
            k_value = None if k_func is None else k_func(nesting)
            return cls(**k_value) if isinstance(k_value, dict) else cls(k_value) if k_value else cls()

        if isinstance(x, dict):
            out = create_instance_k_func()
        elif isinstance(x, tuple) and isinstance(x[0], dict) and len(x) > 1:
            out = cls(**x[1]) if len(x) > 1 else create_instance_k_func()
            x = x[0]
        elif isinstance(x, tuple) and isinstance(x[0], tuple) and len(x) > 1:
            out = cls(**x[1]) if len(x) > 1 else create_instance_k_func()
            x = {k: def_val for k in x[0]}
        elif isinstance(x, tuple) and isinstance(x[0], tuple):
            out = create_instance_k_func()
            x = {k: def_val for k in x[0]}
        elif isinstance(x, tuple):
            out = create_instance_k_func()
            x = {k: def_val for k in x}
        else:
            return x

        keys = out.keys() if out else x.keys()
        for k in keys:
            out[k] = cls.from_nested(x[k], nesting=nesting + (k,), k_func=k_func, def_val=def_val)
        # [out.set_n([k], cls.from_nested(x[k], k_func=k_func, def_val=def_val)) for k in keys]
        return out

    def get_keys_by_value(self, value, invert=False):
        if invert:
            return [k for k, v in self.items() if v != value]
        else:
            return [k for k, v in self.items() if v == value]


    def any(self):
        return self.fn(any, agg_level=0, update=False)

    def all(self):
        return self.fn(all, agg_level=0, update=False)



    def keys_func(self, fn, level=9999, **kwargs):
        def keys_fn(x):
            temp = {fn(k): v for k, v in x.items()}
            x.clear()
            x.update(temp)
            return x
        return self.fn(keys_fn, to_level=True, level=level, on_leaf=False, **kwargs)

    def delete_keys(self, keys, level=9999, **kwargs):
        return self.fn(lambda x: {k: v for k, v in x.items() if k not in keys}, to_level=True, level=level, on_leaf=False, **kwargs)

    def delete_vals(self, vals, level=9999, **kwargs):
        return self.fn(lambda x: {k: v for k, v in x.items() if v not in vals}, to_level=True, level=level, on_leaf=False, **kwargs)

    def delete_vals_by_func(self, fn, level=9999, **kwargs):
        return self.fn(lambda x: {k: v for k, v in x.items() if fn(v)}, to_level=True, level=level, on_leaf=False, **kwargs)

    # def delete_empty_dicts(self, **kwargs):
    #     return self.delete_vals_by_func(lambda x: not isinstance(x, dict) or len(x), **kwargs)

    def find_by_flat_keys_func(self, keys_func):
        flat_self = self.flatten()
        out = self.__class__()
        for k, v in flat_self.items():
            if keys_func(k):
                out.set_n(k, v)

    # def find_by_keys_pattern(self, key_pattern, level=0, search_inside_tuples=False):
    #     if isinstance(key_pattern, str):
    #         def keys_func(x):
    #             return key_pattern in x
    #     elif isinstance(key_pattern, Iterable):
    #         if "**" in key_pattern:
    #             def keys_func(x):
    #                 return me.is_slice_in_list(key_pattern, x)
    #         else:
    #             def keys_func(x):
    #                 return me.is_list_in_list_order(key_pattern, x)


    def find_by_keys(self, keys, level=9999, search_inside_tuples=False):
        out = self.__class__.fromkeys(keys)

        def populate_fn(dictx):
            search_keys = out.get_keys_by_value(None)
            if search_keys:
                if search_inside_tuples:
                    for kx in dictx:
                        for k in out.get_keys_by_value(None):
                            if isinstance(k, (str, Number)):
                                if k in kx:
                                    out[k] = dictx[kx]
                            elif isinstance(k, Iterable):
                                if not set(k).isdisjoint(kx):
                                    out[k] = dictx[kx]
                else:
                    for k in out.get_keys_by_value(None):
                        if isinstance(k, (str, Number)):
                            if k in dictx:
                                out[k] = dictx[k]
                        elif isinstance(k, Iterable):
                            for kx in dictx:
                                if kx in k:
                                    out[k] = dictx[kx]
                return dictx
            else:
                return 'secret_break_keyword777'
        self.fn(populate_fn, to_level=True, level=level, on_leaf=False, update=False)

        final_out = self.__class__()
        for ko in out:
            if isinstance(ko, (str, Number)):
                final_out[ko] = out[ko]
            elif isinstance(ko, Iterable):
                final_out[next(iter(ko))] = out[ko]
            else:
                final_out[ko] = out[ko]
        return final_out

    def find_by_key(self, key, **kwargs):
        return self.find_by_keys([key], **kwargs)[key]

    def cast(self, cast_func, cast_from=None):
        if cast_from is None:
            cast_from = self.__class__
        return self._cast(cast_func, cast_from=cast_from)

    def _cast(self, cast_func, cast_from=None):
        if cast_from is None:
            cast_from = NestedDict
        return cast_func({k: v.cast(cast_func, cast_from=cast_from) if isinstance(v, cast_from) else v
                         for k, v in self.items()})

    def to_dict(self):
        return self.cast(dict)

    def count_levels(self, fn=max):
        return fn(map(len, self.flatten().keys()))

    def _get_best_agg(self, fn=sum, top_n=1, preserve_order=False, reverse=False, on_leaf=False):
        agg = sorted(self.fn(fn, agg_level=1, on_leaf=on_leaf, update=False).items(), key=lambda x: x[1])
        if reverse:
            agg = agg[::-1]
        top_selection = dict(agg[::-1][:top_n])
        if preserve_order:
            out = {k: v for k, v in self.items() if k in top_selection}
        else:
            out = {k: self[k] for k in top_selection}

        return self.__class__(out)

    def get_best_agg(self, *args, level=0, **kwargs):
        return self.fn('_get_best_agg', *args, level=level, fn_kwargs=kwargs)

    def enumerate(self, add=None):
        if add is None:
            return self.__class__.from_dict({i: {k: v} for i, (k, v) in enumerate(self.items())})
        elif isinstance(add, int):
            return self.__class__.from_dict({add + i: {k: v} for i, (k, v) in enumerate(self.items())})
        else:
            return self.__class__.from_dict({f'{add}_{i}': {k: v} for i, (k, v) in enumerate(self.items())})

    def delete_empty(self, condition_func=None, dict_class=None):
        if dict_class is None:
            dict_class = NestedDict
        for k, v in list(self.items()):
            if isinstance(v, dict_class):
                v.delete_empty(condition_func=condition_func, dict_class=dict_class)
                if not v:
                    del self[k]
            else:
                if not condition_func:
                    if not v and not isinstance(v, Number):
                        del self[k]
                else:
                    if condition_func(v):
                        del self[k]
        # [self.pop(k) if not condition_func and not v and not isinstance(v, Number)
        #  or condition_func and not isinstance(v, self.__class__) and condition_func(v)
        #  else v.delete_empty(condition_func=condition_func) for k, v in self.items()
        #  if isinstance(v, self.__class__) or not v]
        return self

    def flatten(self, level=9999, ignore_none=False, key_fn=None, include_branches=False):
        if level == 0:
            return self
        flat = self.__class__()
        levels = range(1, level+1) if include_branches else [level]
        for level_i in levels:
            _flat = {}
            self.fn(lambda x, ind, /: _flat.update({ind: x}), ignore_none=ignore_none, update=False, level=level_i,
                    no_return=True, update_none=True, on_empty_dict=True)
            flat.update({key_fn(k) if key_fn else k: v for k, v in _flat.items()})
            if include_branches and not any(map(lambda x: isinstance(x, NestedDict), _flat.values())):
                break
        return flat

    def levels_keys_dict(self, level=9999):
        out = defaultdict(set)
        for k in self.flatten(level=level).keys():
            for i in range(len(k)):
                out[i+1].add(k[i])
        return dict(out)

        # if level == 0:
        #     return self
        # level -= 1
        # flat = self.__class__()
        # for ks, vs in self.items():
        #     if isinstance(vs, self.__class__) and level:
        #         nes = nesting + (ks,)
        #         print(include_branches)
        #         print(nes)
        #         vs_flat = vs.flatten(level=level, nesting=nes, include_branches=include_branches)
        #         if include_branches:
        #             flat[nes] = vs
        #         for k, v in vs_flat.items():
        #             if ignore_none and v is None:
        #                 continue
        #             flat[(ks,) + k] = v
        #     else:
        #         flat[(ks,)] = vs
        # if key_fn:
        #     flat = self.__class__({key_fn(k): v for k, v in flat.items()})
        # return flat

    def change_nesting(self, copy_objects=False, vec=None, pop_insert=None, outer_in=None, inner_out=None, len_limit=None, level=9999, drop_none=False):
        if self.count_levels() < 2:
            return self.copy()
        main_inputs = [vec, pop_insert, outer_in, inner_out]
        if len_limit is None:
            len_limit = max(map(lambda y: 0 if y is None else (max(map(abs, y)) if isinstance(y, Iterable) else y), main_inputs))
        flat = self.flatten(level=level)
        out = self.copy(deep=False) if copy_objects else self.__class__()
        out.clear()
        once_vec = [True, True, True, True]
        # g_cond = lambda y: y is not None
        def get_g():
            if False:
                pass
            elif vec is not None and once_vec[0]:
                def g(key_len):
                    if len_limit is not None and key_len < len_limit:
                        return None
                    if len(vec) <= key_len:
                        return vec

            elif pop_insert is not None and once_vec[1]:
                def g(key_len):
                    if len_limit is not None and key_len < len_limit:
                        return None
                    _vec = list(range(key_len))
                    ind = _vec.pop(pop_insert[0])
                    at = pop_insert[1]
                    if isinstance(at, str):
                        _vec.append( ind)
                    _vec.insert(at, ind)
                    return _vec

            elif outer_in is not None and once_vec[2]:
                def g(key_len):
                    return list(range(outer_in, key_len)) + list(range(0, outer_in))
            elif inner_out is not None and once_vec[3]:
                def g(key_len):
                    return list(range(key_len-inner_out, key_len)) + list(range(0, key_len-inner_out))
            return g

        g_list = []
        for i, x in enumerate(main_inputs):
            if x is not None:
                g_list.append(get_g())
                once_vec[i] = False
        for k, v in flat.items():
            if not (drop_none and v is None):
                g_vec = list(range(len(k)))
                g_change = False
                for _g in g_list:
                    _ = _g(len(g_vec))
                    if _ is not None:
                        g_change = True
                        g_vec = [g_vec[i] for i in _]
                if not g_change:
                    g_vec = range(k)
                if copy_objects:
                    objects = [self.get_n(p) for p in ef.sequential_permutations(k)]
                    out.set_n([k[i] for i in g_vec], v, objects=[objects[i] for i in g_vec])
                else:
                    out.set_n([k[i] for i in g_vec], v)
        return out

    def copy(self, cast_func=None, deep=True, **kwargs):
        out = deepcopy(self) if deep else _copy(self, cls=self.__class__, **kwargs)
        if cast_func:
            out = cast_func(out)
        elif not isinstance(out, NestedDict):
            return self.__class__(out)
        return out

    def _copy(self, cls=None):
        if cls is None:
            cls = NestedDict
        self_copy = copy(self)
        for k, v in self_copy.items():
            if isinstance(v, cls):
                self_copy[k] = v._copy(cls=cls)
        return self_copy

    def to_list_fn(self, func=None, none_if_missing=False, sort_by_key=False, cast_func=None, level=9999):
        list_out = NestedDictList()

        def func_shell(x, nesting, /):
            for i, y in enumerate(func(x) if func else x):
                if i >= len(list_out):
                    list_out.append(NestedDict().set_n(nesting, y))
                else:
                    list_out[i].set_n(nesting, y)

        self.fn(func_shell, update=False)
        return list_out


class NestedDefaultDict(defaultdict, NestedDict):
    def __init__(self, *args, levels=None, **kwargs):  # func_on_input_vec=None
        self.levels = levels
        self.leaf_method = args[0] if args and callable(args[0]) else None
        if levels is None or isinstance(levels, str):
            args = tuple([lambda: NestedDefaultDict(self.leaf_method, levels=levels)]
                         + [x for x in args if not (callable(x) or x is None)])

        if isinstance(levels, str):
            # if self.leaf_method is not None:
            #     args = args[1:]
            super().__init__(*args, **kwargs)
        elif levels is None or levels <= 1:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(self.custom_factory_method, *args[1:], **kwargs)

    def custom_factory_method(self):
        return self.__class__(self.leaf_method, levels=self.levels - 1)

    # @classmethod
    # def from_flat_dict(cls, flat_dict):
    #     return cls({k: cls.from_nested_dict(v) if isinstance(v, dict) and not isinstance(v, cls)
    #                 else v for k, v in flat_dict.items()})

    def copy(self, cast_func=None):
        out = super().copy()
        out.update({k: v.copy(cast_func=cast_func) if isinstance(v, NestedDict) else v
                    for k, v in out.items()})
        if cast_func:
            out = cast_func(out)
        else:
            if not isinstance(out, NestedDict):
                return self.__class__(out, levels=self.levels)
            out.levels = self.levels
        return out

    # def set_n(self, keys: list, set_val=None):
    #     x = self
    #     if isinstance(self.levels, str) and self.leaf_method is not None:
    #         for k in keys[:-1]:
    #             if k not in x:
    #                 x[k] = self.__class__()
    #             x = x[k]
    #     else:
    #         for k in keys[:-1]:
    #             if not isinstance(x, NestedDict):
    #                 raise KeyError
    #             x = x[k]
    #     if set_val is not None:
    #         x[keys[-1]] = set_val
    #     elif isinstance(self.levels, str) and self.leaf_method is not None:
    #         x[keys[-1]] = self.leaf_method()
    #     return x[keys[-1]]


class NestedDictList(list):

    def iter(self):
        return enumerate(self)

    def __init__(self, seq=(), init_dict=False, **kwargs):  # func_on_input_vec=None
        """
        :param seq: list of dicts or NestedDicts
        :param init_dict: if True, will initialize each element as a NestedDict, set True if seq is a list of dicts
        :param kwargs: kwargs for NestedDict constructor
        """
        if init_dict:
            super().__init__(map(lambda x: NestedDict(x, **kwargs), seq))
        else:
            super().__init__(seq)

    def flatten(self, use_inner_cls=False, **kwargs):
        flat, outer = ef.flatten(self, NestedDictList, return_outer=True)
        return outer.__class__(flat) if use_inner_cls else self.__class__(flat)

    def concat(self):
        return self.list_fn()

    def copy(self, cast_func=None, deep=True):
        out = deepcopy(self) if deep else copy(self)
        for i, x in enumerate(out):
            if cast_func:
                out[i] = cast_func(x)
        return out

    def func(self, func, *args, **kwargs):
        return self.__class__([func(x, *args, **kwargs) for x in self])

    def func_with_other(self, func, other, *args, **kwargs):
        return self.__class__([func(x, other[i], *args, **kwargs) for i, x in enumerate(self)])

    def fn(self, func, *args, **kwargs):
        return self.__class__([x.fn(func, *args, **kwargs) for x in self])

    def list_fn(self, func=None, none_if_missing=False, equal_keys=False, sort_by_key=False, cast_func=None, level=9999):
        """

        """
        if len(self) == 0:
            return None
        if not isinstance(self[0], NestedDict):
            return func(self)

        flats = [x.flatten(level=level) for x in self]
        flats_lens = [len(x) for x in flats]
        out = self[flats_lens.index(max(flats_lens))].copy()
        flat_merged_keys = {}
        # [flat_merged_keys.update(dict.fromkeys(x.keys())) for x in flats]
        if equal_keys:
            flat_merged_keys.update(dict.fromkeys(flats[0].keys()))
        else:
            [flat_merged_keys.update(dict.fromkeys(x.keys())) for x in flats]

        for key_tuple in flat_merged_keys:
            lis = []
            for fl in flats:
                if key_tuple in fl:
                    lis.append(fl[key_tuple])
                elif none_if_missing:
                    lis.append(None)
            if func:
                out.set_n(list(key_tuple), func(lis))
            else:
                out.set_n(list(key_tuple), lis)

        return out



def dict_tuple_list_recursive(x, func, pass_keys=None):
    if isinstance(x, dict):
        if isinstance(pass_keys, (tuple, list)):
            pass_keys = list(pass_keys)
        return {k: dict_tuple_list_recursive(v, func, pass_keys=pass_keys + [k]) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple([dict_tuple_list_recursive(v, func, pass_keys=pass_keys) for v in x])
    elif isinstance(x, list):
        return [dict_tuple_list_recursive(v, func, pass_keys=pass_keys) for v in x]
    else:
        if pass_keys:
            return func(x, pass_keys)
        else:
            return func(x)

#
# class NestedUtils:
#
#     def iter(self):
#         raise NotImplementedError
#
#
#
#     # def construct_fn(self):
#     #     base_type = type(self)
#
#     def fn(self, func, *args, replace_keys=True, update=True, update_none=False, on_leaf=None,
#            ignore_keys=None, keep_ignored=False, discard_none=False,
#            func_n_args=None, on_empty_dict=False, nesting=(), cast_func=None,
#            no_return=False,  cast_nested=False, ignore_none=True,
#            level=None, to_level=False, parent=None,
#            agg_level=None, agg_fn=None, agg_chain_iterable=True,
#            fn_args=None, fn_kwargs=None, **kwargs):
#         """
#         @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#         WHEN ADDING KEYWORD ARGUMENTS, ADD THEM TO KW BELOW AS WELL
#         except: nesting, parent, level
#         @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#         """
#         base_type = type(self)
#         val_fn = (lambda x: x.values()) if base_type == dict else (lambda x: x)
#         iter_fn = (lambda x: x.items()) if base_type == dict else (lambda x: enumerate(x))
#
#         if level is None:
#             level = 9999
#         if agg_level is None:
#             agg_level = 9999
#         if agg_level < 0:
#             agg_level = self.count_levels() + agg_level
#             if agg_level < 0:
#                 warnings.warn('abs(agg_level) is more than the number of levels in the dict, agg_level set to 0')
#                 agg_level = 0
#         elif agg_level > 0:
#             agg_level = agg_level - 1
#         if on_leaf is None and agg_level > 1000:
#             on_leaf = True
#             on_leaf_pass = None
#         else:
#             on_leaf_pass = on_leaf
#
#         if fn_args is None:
#             fn_args = tuple()
#         if fn_kwargs is None:
#             fn_kwargs = {}
#         if cast_nested:
#             cast_func = self.__class__.from_dict
#
#         if no_return:
#             update = False
#             cast_func = dict
#
#         kw = dict(replace_keys=replace_keys, update=update, update_none=update_none, on_leaf = on_leaf_pass,
#         ignore_keys=ignore_keys, keep_ignored=keep_ignored, discard_none=discard_none,
#         func_n_args=func_n_args, on_empty_dict=on_empty_dict, cast_func=cast_func,
#         ignore_none=ignore_none,
#         to_level=to_level,
#         agg_level=agg_level, agg_fn=agg_fn, agg_chain_iterable=agg_chain_iterable,
#         fn_args=fn_args, fn_kwargs=fn_kwargs, **kwargs)
#
#         inst_method = isinstance(func, str)
#         func4sig = getattr(self, func) if inst_method else func
#         try:
#             pos_param = sum(1 for p in signature(func4sig).parameters.values() if p.kind == p.POSITIONAL_ONLY)
#             pos_or_key_param = sum(1 for p in signature(func4sig).parameters.values() if p.kind in {
#                 p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL})
#             if inst_method:
#                 sig = max(pos_param -1, 0)
#             else:
#                 sig = pos_param or pos_or_key_param
#         except ValueError:
#             pos_param = 0
#             if inst_method:
#                 sig = 0
#             else:
#                 sig = 1
#         if func_n_args is not None:
#             sig = func_n_args
#
#         if inst_method:
#             func_str = func
#             if sig == 0:
#                 _func = lambda x, *_args: getattr(x, func_str)(*args, *fn_args, **fn_kwargs, **kwargs) if isinstance(x, NestedDict) else x
#             if sig >= 1:
#                 _func = lambda x, *_args: getattr(x, func_str)(*_args, *args, *fn_args, **fn_kwargs, **kwargs) if isinstance(x, NestedDict) else x
#         else:
#             if sig == 0:
#                 _func = lambda x, *_args: func(*args, *fn_args, **fn_kwargs, **kwargs)
#             elif pos_param == 2:
#                 _func = lambda x, *_args: func(x, _args[0], *args, *fn_args, **fn_kwargs, **kwargs)
#             elif pos_param >= 3:
#                 _func = lambda x, *_args: func(x, *_args, *args, *fn_args, **fn_kwargs, **kwargs)
#             elif sig >= 1:
#                 _func = lambda x, *_args: func(x, *args, *fn_args, **fn_kwargs, **kwargs)
#
#
#         empty_flag = on_empty_dict and len(self) == 0
#         out = _func(self, nesting, parent) if empty_flag or to_level or level == 0 else self
#         out = out if out is not None else self
#         if out == 'secret_break_keyword777':
#             return None
#         if out is None:
#             return None
#         if level != 0 and not empty_flag:
#             if level > 0:
#                 level = max(0, level - 1)
#             out_temp = {}
#             for k, v in out.items():
#                 if ignore_keys is not None and k in ignore_keys:
#                     if keep_ignored:
#                         out_temp[k] = v
#                     continue
#                 apply_fn_flag = False
#                 if isinstance(v, NestedDict) and level < 0:
#                     if v.count_levels() == -level:
#                         apply_fn_flag = True
#                 if not apply_fn_flag and isinstance(v, NestedDict) and (level or to_level):
#                     out_temp[k] = v.fn(func, *args, nesting=nesting + (k,), parent=self, level=level, **kw)
#                 else:
#                     if apply_fn_flag or (isinstance(v, NestedDict) or on_leaf) and not to_level and not (ignore_none and v is None):
#                         temp = _func(v, nesting + (k,), self)
#                         out_temp[k] = temp if not update or update and (update_none or temp is not None) else v
#                     else:
#                         out_temp[k] = v
#                     if discard_none and out_temp[k] is None:
#                         del out_temp[k]
#             # out = {k: v.fn(func, *args, nesting=nesting + (k,), level=level, **kw, **kwargs)
#             #        if isinstance(v, self.__class__) and (level or to_level)
#             #        else (_func(v, nesting + (k,))
#             #              if (isinstance(v, self.__class__) or on_leaf) and not to_level and not (ignore_none and v is None)
#             #              else v)
#             #        for k, v in out.items()}
#             if out is self:
#                 out = out_temp
#             else:
#                 out.clear()
#                 out.update(out_temp)
#         if agg_level == 0 and not empty_flag:
#             out_vals = val_fn(out)
#             if agg_chain_iterable and all(map(lambda x : isinstance(x, Iterable), out_vals)):
#                 out_vals = it.chain(*out_vals)
#             if agg_fn is None:
#                 return _func(list(out_vals), nesting)
#             else:
#                 if callable(agg_fn) and not agg_fn == list :
#                     return agg_fn(list(out_vals))
#                 else:
#                     return list(out_vals)
#
#
#         if out is self:
#             return self
#         if cast_func is None:
#             if isinstance(out, base_type):
#                 out = self.__class__(out)
#         else:
#             out = cast_func(out)
#         if update:
#             # if update_none or out is not None:
#             if replace_keys:
#                 self.clear()
#             self.update(out)
#             return self
#         else:
#             return out



# # NestedDict.from_dict({'A':{'B':{'C':{'D':{'D':{'D':5}}}, 'B':{'s':9} }}, 'q':66}).get_best_agg(level=0, top_n=2)
# NestedDict.from_dict({'A':{'B':{'C':{'D':{'D':{'D':5}}}, 'B':{'s':9} }}, 'q':66}).get_best_agg(level=2, top_n=4)
# NestedDict.from_dict({'A':{'B':{'C':{'D':{'D':{'D':5}}}, 'B':{'s':9} }}, 'q':66}).get_best_agg(level=1)
# NestedDict.from_dict({'A':{'B':{'C':{'D':5}, 'B':{'s':9} }}}).keys_func(lambda k: k.lower() if isinstance(k, str) else k)
# # # k.lower() if isinstance(k, str) else k
# # # NestedDict.from_dict().fn(lambda x: {fn: v for k, v in x.items()}, to_level=True, update=True, on_leaf=False)
# # NestedDict.from_dict({'a':{'b':{'c':{'d':8}, 'b':{'d':8} }}}).fn(print, level=0)