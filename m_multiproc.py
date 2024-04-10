import multiprocessing as mp
import threading


def thread_concurrently(func, args, args_tuples=False, results=None):
    threads = []

    for x in args:
        x_pass =  list(x) if args_tuples else [x]
        if results is not None:
            x_pass.append(results)
        thread = threading.Thread(target=func, args=x_pass)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    if results is not None:
        return results


import multiprocessing as mp


# def mp_concurrently(func, args, args_tuples=False, results=None):
#     def worker(init_args):
#         if args_tuples:
#             func(*init_args)
#         else:
#             func(init_args)
#
#     with mp.get_context("spawn").Pool() as pool:
#         pool.map(worker, args if args_tuples else [(arg,) for arg in args])


def mp_concurrently(func, args, args_tuples=False, results=None):
    def coerce_x(x):
        x_pass = list(x) if args_tuples else [x]
        if results is not None:
            x_pass.append(results)
        return x_pass
    with mp.get_context("spawn").Pool() as pool:
        res_out = pool.map(func, list(map(coerce_x, args)))

    if results is not None:
        return results
    else:
        return res_out

# if __name__ == "__main__":
#     main()