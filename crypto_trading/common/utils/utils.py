from typing import List


def chunk_list(lst: List, n: int) -> List:
    """
    Breaks up a list into smaller chunks of length n.

    Args:
    lst (list): The input list to be chunked.
    n (int): The desired chunk size.

    Returns:
    list: A list of smaller lists, each of length n (except the last one which may be shorter).
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]
