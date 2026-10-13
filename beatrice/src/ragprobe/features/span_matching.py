def find_subsequence(full_ids, sub_ids):
    n = len(sub_ids)
    if n == 0:
        return -1, -1

    for i in range(len(full_ids) - n + 1):
        if full_ids[i:i + n] == sub_ids:
            return i, i + n

    if n > 2:
        relaxed = sub_ids[1:-1]
        m = len(relaxed)
        for i in range(len(full_ids) - m + 1):
            if full_ids[i:i + m] == relaxed:
                return i, i + m

    if n > 4:
        relaxed = sub_ids[2:-2]
        m = len(relaxed)
        for i in range(len(full_ids) - m + 1):
            if full_ids[i:i + m] == relaxed:
                return i, i + m

    return -1, -1
