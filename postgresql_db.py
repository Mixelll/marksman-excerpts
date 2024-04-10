"""
This module is intended for the creation and execution of `psycopg3`
composables, representing queries-pending-values. The composables are further
composed together to produce PostgreSQL queries. It leverages nesting, parsing
and keywords to generate complex queries with properly escaped names.

Parameters are denoted by :param_or_type:, brackets in conjunction with this
notation have the same meaning as when used as a Python literal.

The list of major parameters recurring in the module:

:param columns:
    :[str or [str]]: List of strings[] representing column names
    passed to `composed_columns`.

:param schema:
    :str: representing schema name

:param returning:
    :[str or NestedIterable]: parsable by `composed_parse` representing
    returned expressions.

:param conflict:
    :[str]: representing column names following SQL's `ON CONFLICT`

:param nothing:
    Wether to `DO NOTHING` following SQL's `ON CONFLICT`

:param set:
    :[str: or Iterable]:, parsables passed to `composed_set`

:param parse:
    :Callable: passed to `composed_set` and `composed_separated`
    as `parse=parse`, usually `composed_parse` is used.

:param where:
    :[str or NestedIterable]:, parsable passed to `composed_parse`
"""






# from inspect import getmembers, isfunction, isclass
import io
import pandas as pd
import psycopg as psg
import sqlalchemy as sqa

import extra_functions as me

from psycopg import sql

from uuid import uuid4
from credentials import db_trades

# execute connection objects
# with open('connections.txt') as f:
#     exec(f.read())
# conn
# engine

def connect(dbname=db_trades.dbname, user=db_trades.user, password=db_trades.password):
    return psg.connect(dbname=dbname, user=user, password=password)

def create_engine(dbname=db_trades.dbname, user=db_trades.user, password=db_trades.password):
    return sqa.create_engine(f'postgresql+psycopg://{user}:{password}@localhost:5432/{dbname}')




coerce = lambda x, typ: x if isinstance(x, typ) else typ(x)

def c_(x, typ):

    def create(y):
        if isinstance(y, (tuple, list)):
            return typ(*y[1:]) if len(y) >= 2 and y[0] is None else typ(*y)
        else:
            return typ(y)

    return x if isinstance(x, typ) else create(x)


S = sql.SQL
I = sql.Identifier
P = sql.Placeholder()


def I_2(x):
    return c_(x, I)


def psg_operators():
    return S, I, P, I_2

def get_conn_if_engine(conn_or_engine):
    if all(map(lambda x: x in str(type(conn_or_engine)).lower(),
               ['sqlalchemy', 'engine'])):
        return conn_or_engine.raw_connection()
    else:
        return conn_or_engine


def vacuum_table(conn_or_engine, table, schema=None, analyze=False):
    conn = get_conn_if_engine(conn_or_engine)
    if schema:
        table = schema, table
    if analyze:
        query = S('VACUUM FULL ANALYZE {}').format(I_2(table))
    else:
        query = S('VACUUM FULL {}').format(I_2(table))
    pg_execute(conn, query, commit=True)


approvedExp = ['primary key', 'foreign key', 'references', 'default', 'uuid',
                'on delete restrict', 'unique', 'timestamp with time zone',
                'on delete cascade', 'current_timestamp','double precision',
                'not null', 'json', 'bytea', 'timestamp', 'like', 'and',
                'or', 'join', 'inner', 'left', 'right', 'full', '', '.', ',',
                 '=', '||','++','||+', 'min', 'max', 'least', 'greatest', 'like',
                '/']


def query_get_df(conn_or_engine, query, index=None, psg_params=None):
    if 'sqlalchemy' in str(type(conn_or_engine)).lower():
        if isinstance(query, sql.Composable):
            query = pg_execute(conn_or_engine.raw_connection(), query, params=psg_params, mogrify_return=True)
        return pd.read_sql_query(query, con=conn_or_engine, index_col=index)
    else:
        df = pd.DataFrame(pg_execute(conn_or_engine, query, params=psg_params))
        df.set_index(index, inplace=True)
        return df


def pg_execute(conn_obj, query, params=None, commit=True, autocommit=False, as_string=False, mogrify_print=False, mogrify_return=False):
    """
    Execute a psycopg2 composable, possibly containing a placeholder -
    *sql.Placeholder* or `'%s'` for the `params`.

    :param conn_obj:
        :psycopg.connection: or :sqlalchemy.engine.base.Engine:,
        connection object to execute the query with

    :param query:
        :Composable:, query awaiting params

    :param params:
        :[str or NestedIterable]:, params to be passed to the query

    :param as_string:
        Whether to print the output of `conn.cursor.as_string(...)`

    :param mogrify_print:
        Whether to print the output of `conn.cursor.mogrify(...)`

    :param mogrify_return:
        Whether to return the output of `conn.cursor.mogrify(...)`

    :return:
        :[(...)]:, returns array of tuples (rows)
    """
    conn = get_conn_if_engine(conn_obj)
    # conn.autocommit = autocommit
    mogrify = mogrify_print or mogrify_return
    cur = psg.ClientCursor(conn) if mogrify else conn.cursor()
    # cur = psg.ClientCursor(conn)
    if as_string:
        print(query.as_string(conn))
    if mogrify:
        mog = cur.mogrify(query, params)
        if mogrify_print:
            print(mog)
        if mogrify_return:
            return mog
    cur.execute(query, params)
    if commit:
        conn.commit()
    if cur.description is not None:
        return cur.fetchall()


def composed_columns(columns, enclose=False, parse=None, literal=None, **kwargs):
    s = psg_operators()[0]
    if parse is None:
        parse = lambda x: composed_separated(x, '.', **kwargs)
    if isinstance(columns, str):
        columns = [columns]

    if columns is None:
        return s('*')
    else:
        comp = s(', ').join(map(parse, columns))
        if enclose:
            return s('({})').format(comp)
        return comp


def composed_parse(exp, safe=False, tuple_parse=composed_columns):
    """
    Parse a nested container of strings by recursively pruning according
    to the following rules:

    - Enclose expression with `'$'` to parse the string raw into the quary,
        only selected expressions are allowed. *
    - If exp is  `'%s'` or *sql.Placeholder* to parse into *sql.Placeholder*.
    - If exp is a tuple it will be parsed by `composed_columns`.
    - If exp is a dict the keys will be parsed by `composed_columns` only if
        exp[key] evaluates to True.
    - Else (expecting an iterable) `composed_parse` will be applied to
        each element in the iterable.

    :param safe:
        Wether to disable the raw parsing in *

    :param exp:
        :[:str: or :Iterable:] or str: List (or string) of parsables
        expressions (strings or iterables).

    :param enclose:
        :Bool: passed to `composed_columns` as `enclose=enclose`

    :param parse:
        :Callable: passed to `composed_columns` as `parse=parse`,
        usually `composed_parse` itself is used.

    :return:
        :Composable:
    """

    if isinstance(exp, str):
        if not safe and exp[0]=='$' and exp[-1]=='$':
            exp = exp.replace('$','')
            if exp.strip(' []()').lower() in approvedExp:
                returned =  S(exp)
            elif exp.strip(' []()') == '%s':
                e = exp.split('%s')
                returned = S(e[0]) + P + S(e[1])
            else:
                raise ValueError(f'Expression: {exp.strip(" []()")} not found in allowed expressions')
        elif exp.strip('$ []()') == '%s':
            e = exp.replace('$','').split('%s')
            returned = S(e[0]) + P + S(e[1])
        else:
            returned =  I(exp)
    elif isinstance(exp, sql.Placeholder):
        returned = exp
    elif isinstance(exp, tuple):
        returned = tuple_parse(filter(me.mbool, exp))
    elif isinstance(exp, dict):
        returned = tuple_parse(filter(me.mbool, [k for k in exp.keys() if exp[k]]))
    else:
        expPrev = exp[0]
        for x in exp[1:]:
            if x == expPrev:
                raise ValueError(f"Something's funny going on - {x,x} pattern is repeated ")
            else:
                expPrev = x

        return sql.Composed([composed_parse(x, safe=safe, tuple_parse=tuple_parse) for x in filter(me.mbool, exp)])

    return S(' {} ').format(returned)


def composed_insert(tbl, columns, returning=None, conflict=None,
                    nothing=False, set=None, parse=composed_parse):
    """
    Construct query with value-placeholders to insert a row into `"tbl"` or `"tbl[0]"."tbl[1]"`

    :return:
        :Composable:, query awaiting params
    """
    comp = S('INSERT INTO {} ({}) VALUES ({}) ').format(I_2(tbl), composed_separated(columns),
                                                        composed_separated(len(columns) * [P]))

    if conflict:
        if nothing:
            comp += S('ON CONFLICT ({}) DO NOTHING').format(I(conflict))
        else:
            if set is None:
                set = columns
            comp += S('ON CONFLICT ({}) DO UPDATE').format(I(conflict)) + composed_set(set, parse=parse)

    if returning is not None:
        comp += S(' RETURNING {}').format(composed_separated(returning, parse=parse))

    return comp


def composed_update(tbl, columns, returning=None, where=None,
                    parse=composed_parse):
    """
    Construct query with value-placeholders to insert a row into `"schema"."tbl"`

    :return:
        :Composable:, query awaiting params
    """

    comp = S('UPDATE {}').format(I_2(tbl)) + composed_set(columns)

    if where is not None:
        comp += S(' WHERE {}').format(parse(where))

    if returning is not None:
        comp += S(' RETURNING {}').format(parse(returning))

    return comp


def composed_create(tbl, columns, schema=None, like=None,
                    inherits=None, constraint=None, parse=composed_parse):
    """
    Create a table as `"schema"."tbl"`

    :param like:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param inherits:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param constraint:
        :[str or NestedIterable]:, table constraints
        parsable by `composed_parse`, passed to `composed_columns`

    :return:
        :Composable:, full create table query
    """

    if isinstance(columns[0], str):
        columns = [columns]
    comp = S('CREATE TABLE {}{} (').format(composed_dot(schema), I(tbl))

    if like is not None:
        comp += composed_columns(like, parse=lambda x:
            S('LIKE {} INCLUDING ALL, ').format(composed_separated(x, '.', )))

    if constraint:
        if isinstance(constraint[0], str): constraint = [constraint]
        comp += composed_columns(constraint, parse=lambda x:
            S('CONSTRAINT ({}), ').format(parse(x)))

    comp += composed_columns(columns, parse=parse) + S(') ')

    if inherits is not None:
        comp += S('INHERITS ({})').format(composed_columns(inherits))

    return comp

def composed_where(index):
    return S('WHERE {} ').format(I(index))

def composed_select_from_table(tbl, columns=None, where_between=None, params=None):
    """
    Select columns from table as `"schema"."tbl"`

    :return:
        :Composable:, full select query which can be further used to compose
    """
    query = S('SELECT {} FROM {} ').format(composed_columns(columns),I_2(tbl))
    if where_between is not None:
        between = where_between[1]
        btw = composed_between(start=between[0], end=between[1])
        query += composed_where(where_between[0]) + btw[0]
        params.extend(btw[1])

    return query


def composed_from_join(join=None, tables=None, columns=None, using=None, parse=composed_parse):
    # def I_2(x): composed_separated(x, '.')

    joinc = []
    for v in multiply_iter(join, max(iter_length(tables, columns, using))):
        vj = '$'+v+'$' if v else v
        joinc.append(parse([vj, '$ JOIN $']))

    if tables:
        tables = list(tables)
        comp = S('FROM {} ').format(I_2(tables[0]))
        if using:
                for t, u, jo in zip(tables[1:], using, joinc):
                    comp += jo + S('{} USING ({}) ').format(I_2(t), composed_columns(u))
        elif columns:
            for t, co, jo in zip(tables[1:], columns, joinc):
                comp += jo + S('{} ').format(I_2(t))
                for j, c in enumerate(co):
                    comp += S('ON {} = {} ').format(I_2(c[0]), I_2(c[1]))
                    if j < len(co):
                        comp += S('AND ')
        else:
            for t in tables[1:]:
                comp += S('NATURAL ') + parse([join, '$ JOIN $']) + S('{} ').format(I_2(t))

    elif columns:
        columns = list(columns)
        comp = S('FROM {} ').format(I_2(columns[0][:-1]))
        for i in range(1, len(columns)):
            toMap = columns[i][:-1], columns[i-1], columns[i-1]
            comp += joinc[i-1] + S('{} ON {} = {} ').format(*map(I_2, toMap))
    else:
        raise ValueError("Either tables or columns need to be given")

    return comp


def composed_set(set_obj, parse=composed_parse):
    """
    Return a composable of the form `SET (...) = (...)`

    :param like:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param inherits:
        :[str or [str]]:, parsables passed to `composed_columns`

    :param set_obj:
        :[str or NestedIterable]:, set table columns
        parsable by `composed_parse`, passed to `composed_columns` and
         `composed_separated`

    :return:
        :Composable:
    """
    if not set_obj:
        return S('')
    col, val = [], []
    for c in set_obj:
        if isinstance(c, (tuple, list)):
            if len(c)>1:
                col.append(c[0])
                val.append(c[1:])
            else:
                col.append(c[0])
                val.append(P)
        else:
                col.append(c)
                val.append(P)
    if len(col)>1:
        formatted = S(' SET ({}) = ({})')
    else:
        formatted = S(' SET {} = {}')
    return formatted.format(composed_columns(col),
                            composed_separated(val, parse=parse))


def composed_between(start=None, end=None):
    """
    Return a composable that compares values to `start` and `end`

    :param start:
        :str or datetime or numeric:

    :param end:
        :str or datetime or numeric:

    :return:
        :(Composable, Array):, composable and values passed to `pg_execute` are returned
    """
    s = psg_operators()[0]
    comp = s('')
    execV = []

    if start is not None and end is not None:
        comp += s('BETWEEN %s AND %s ')
        execV.extend([start, end])
    elif start is None and end is not None:
        comp += s('<= %s ')
        execV.append(end)
    elif start is not None:
        comp += s('>= %s ')
        execV.append(start)

    return comp, execV


def composed_dot(name):
    if name:
        if not isinstance(name, str):
            return [composed_dot(x) for x in name]
        return S('{}.').format(I(name))
    return S('')



def composed_separated(names, sep=', ', enclose=False, AS=False, parse=None):
    if parse is None:
        parse = composed_parse
    if isinstance(names, str):
        names = [names]
    names = list(filter(me.mbool, names))
    if sep in [',', '.', ', ', ' ', '    ']:
        comp = S(sep).join(map(parse, names))
        if AS:
            comp += S(' ') + N(sep.join(names))
        if enclose:
            return S('({})').format(comp)
        return comp
    else:
        raise ValueError(f'Expression: "{sep}" not found in approved separators')


def append_df_to_db(engine, ident, df, index=True):
    schema, tbl = ident
    conn = engine.raw_connection()
    df.head(0).to_sql(tbl, engine, if_exists='append', index=index, schema=schema)
    cur = conn.cursor()
    copy_df2db(cur, ident, df, index=index)
    conn.commit()


def copy_df2db(cur, df, ident, index=True):
    output = io.StringIO()
    df.to_csv(output, sep='\t', header=True, index=index)
    output.seek(0)
    ident = I_2(ident)
    with output as f:
        with cur.copy(S("COPY {} FROM STDIN DELIMITER '\t' CSV HEADER;").format(ident)) as copy:
            while data := f.read(100):
                copy.write(data)


def upsert_df_to_db(engine, ident, df, index=True):
    schema, tbl = ident
    df.head(0).to_sql(tbl, engine, if_exists='append', index=index, schema=schema)
    conn = engine.raw_connection()
    cur = conn.cursor()
    temp = I(tbl + '_' + str(uuid4())[:8])
    ident = I_2(ident)
    cur.execute(S('CREATE TEMP TABLE {} (LIKE {} INCLUDING ALL);').format(temp, ident))
    copy_df2db(cur, df, temp, index=index)
    # cur.copy(s("COPY {} FROM STDIN DELIMITER '\t' CSV HEADER;").format(temp), output)
    cur.execute(S('DELETE FROM {} WHERE ({index}) IN (SELECT {index} FROM {});')
        .format(ident, temp, index=composed_separated(tuple(df.index.names))))
    cur.execute(S('INSERT INTO {} SELECT * FROM {};').format(ident, temp))
    cur.execute(S('DROP TABLE {};').format(temp))
    conn.commit()

# def create_trigger(name, ident, func, **kwargs):
#
#     comp = S("""
#     CREATE OR REPLACE TRIGGER {}
#     AFTER TRUNCATE ON {}
#     FOR EACH STATEMENT
#     EXECUTE FUNCTION {}();
#     """).format(c_(name, I), c_(ident, I), c_(func, I))
#     return comp
#
# def ident2name(x, **kwargs):
#     if isinstance(x, (list, tuple)):
#         return '_'.join(x)
#     elif isinstance(x, str):
#         return x
#     else:
#         return str(uuid4())

def get_tableNames(conn, names, operator='like', not_=False, relkind=('r', 'v'),
                    case=False, schema=None, qualified=None):
    s = psg_operators()[0]
    relkind = (relkind,) if isinstance(relkind, str) else tuple(relkind)
    c, names = composed_regex(operator, names, not_=not_, case=case)
    execV = [relkind, names]
    if schema:
        execV.append((schema,) if isinstance(schema, str) else tuple(schema))
        a = s('AND n.nspname IN %s')
    else:
        a = s('')

    cursor = conn.cursor()
    cursor.execute(s('SELECT {} FROM pg_class c JOIN pg_namespace n ON \
                    n.oid = c.relnamespace WHERE relkind IN %s AND relname {} %s {};') \
        .format(composed_parse({'nspname': qualified, 'relname': True}, safe=True), c, a), execV)
    if qualified:
        return cursor.fetchall()
    else:
        return [x for x, in cursor.fetchall()]


def exmog(cursor, input):
    print(cursor.mogrify(*input))
    cursor.execute(*input)

    return cursor


def composed_regex(operator, names, not_, case):
    s = psg_operators()[0]
    if operator.lower() == 'like':
        c = s('LIKE') if case else s('ILIKE')
        c = s('NOT ')+c+s(' ALL') if not_ else c+s(' ANY')
        if isinstance(names, str):
            names = [names]
        names = (names,)
    elif operator.lower() == 'similar':
        c = s('NOT SIMILAR TO') if not_ else s('SIMILAR TO')
        if not isinstance(names, str):
            names = '|'.join(names)
    elif operator.lower() == 'posix':
        c = s('~')
        if not case:
            c += s('*')
        if not_:
            c = s('!') + c
        if not isinstance(names, str):
            names = '|'.join(names)

    return c, names


def table_exists(conn, name):
    exists = False
    try:
        cur = conn.cursor()
        cur.execute(f"select exists(select relname from pg_class where relname='{name}')")
        exists = cur.fetchone()[0]
        cur.close()
    except psg.Error as e:
        print(e)

    return exists


def get_table_column_names(conn, name):
    conn = get_conn_if_engine(conn)
    column_names = []
    try:
        cur = conn.cursor()
        cur.execute(S('select * from {} LIMIT 0').format(I_2(name)))
        for desc in cur.description:
            column_names.append(desc[0])
        cur.close()
    except psg.Error as e:
        print(e)

    return column_names

# def add_table_column(conn, name, column_name, column_type):
#     conn = get_conn_if_engine(conn)
#     try:
#         comp = S('ALTER TABLE {} ADD COLUMN {} {};').format(I_2(name), I_2(column_name), column_type)
#         pg_execute(conn, comp)
#     except psg.Error as e:
#         print(e)
#
#
# def add_table_column_by_values(conn, name, column_dict):
#     pd.DataFrame(column_dict).to_sql(name, conn, if_exists='append', index=False)
#
#
# def add_table_columns_if_not_existing(conn, name, column_name, column_type):
#     conn = get_conn_if_engine(conn)
#     if column_name not in get_table_column_names(conn, name):
#         add_table_columns(conn, name, column_name, column_type)




def set_comment(conn, tbl, comment, schema=None):
    schema = composed_dot(schema)
    query = S('COMMENT ON TABLE {}{} IS %s').format(schema, I(tbl))
    return pg_execute(conn, query, params=[str(comment)])
