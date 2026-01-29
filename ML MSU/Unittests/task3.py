def find_modified_max_argmax(L, f):
    v = [f(x) for x in L if type(x) == int]
    if not v:
        return ()
    m = max(v)
    i = v.index(m)
    return m, i
