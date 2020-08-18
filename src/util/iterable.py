from itertools import chain, repeat, accumulate, tee


def chunk(xs, n):
    """
    Split list into n chunks.
    """

    assert n > 0
    length = len(xs)
    s, r = divmod(length, n)
    widths = chain(repeat(s + 1, r), repeat(s, n - r))
    offsets = accumulate(chain((0,), widths))
    b, e = tee(offsets)
    next(e)
    return [xs[s] for s in map(slice, b, e)]


def split_into_batches(things, batch_size):
    """
    Split list into minibatches.
    """

    for i in range(0, len(things), batch_size):
        batch = things[i:min(i + batch_size, len(things))]
        if len(batch) == batch_size:
            yield batch
