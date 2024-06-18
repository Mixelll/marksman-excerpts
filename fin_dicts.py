from special_dicts import NestedDict, NestedDictList
import extra_functions as ef


class FinDict(NestedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class TimeSeriesDict(FinDict):
    def __init__(self, *args, df=None, df_index=None, index_fn=None, t=None, t_cls=None, stats=None, ds=None, model=None, models=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df = df
        self.df_index = df_index
        self.t = t
        self.index_fn = index_fn
        self.stats = stats
        self.ds = ds
        self.model = model
        self.models = models
        self.t_cls = t_cls

        if self.t_cls is None:
            if self.t is not None:
                self.t_cls = self.t.__class__
            else:
                self.t_cls = tuple

        if self.df is not None:
            if self.index_fn is None:
                self.index_fn = lambda x: x.df.index
            if self.df_index is None:
                self.reindex()

        if self.models is None:
            self.models = {}
        if self.stats is None:
            self.stats = {}

        if t is None and df_index is not None:
            self.t = (self.df_index[0], self.df_index[-1])

    @classmethod
    def from_df(cls, df, *args, **kwargs):
        out = cls(df=df, df_index=df.index)
        out.recursive_op(*args, **kwargs)
        return out

    def recursive_op(self, oper_struct, oper_dict=None):
        # [f(self) for f in operations[0]]
        fn = lambda x, o: x.fn('recursive_op',o, oper_dict=oper_dict, update=False, on_empty_dict=True)
        if isinstance(oper_struct, list):
            oper_struct[0](self)
            if oper_struct[1:]:
                fn(self, oper_struct[1:])
                # [out[k].recursive_op(oper_struct[1:]) for k in out.keys()]
        elif isinstance(oper_struct, dict):
            for k, v in oper_struct.items():
                set_i = set(self.keys())
                oper_dict[k](self)
                [fn(d[x], v) for x in set(self.keys()) - set_i]
        # return self


    def reindex(self, t=None):
        if t is not None:
            self.t = t
        self.df_index =  self.index_fn(self)
# class TimeSeriesDict(FinDict):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     @classmethod
#     def from_df(cls, df, time_quants):
#         self.data = df
#
#         return cls({k: cls.from_dict(v, levels) if isinstance(v, dict) and not isinstance(v, cls) and levels else v
#                     for k, v in nested_dict.items()})