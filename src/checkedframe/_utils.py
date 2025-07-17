from __future__ import annotations

import inspect
import types
from collections.abc import Iterable
from typing import Any


# Based on the inspect module. We can't use the inspect version directly because it
# sorts the members by name, with no option to turn this off. We need an unsorted
# version because the order of columns in the schema may matter.
def get_class_members(object, predicate=None):
    """Return all members of an object as (name, value) pairs sorted by name.
    Optionally, only return members that satisfy a given predicate."""
    assert inspect.isclass(object)

    mro = (object,) + inspect.getmro(object)
    results = []
    processed = set()

    names = []
    for base in reversed(mro):
        for key in base.__dict__:
            if key not in processed:
                names.append(key)

            processed.add(key)
    # :dd any DynamicClassAttributes to the list of names if object is a class;
    # this may result in duplicate entries if, for example, a virtual
    # attribute with the same name as a DynamicClassAttribute exists
    try:
        for base in object.__bases__:
            for k, v in base.__dict__.items():
                if isinstance(v, types.DynamicClassAttribute):
                    names.append(k)
    except AttributeError:
        pass
    for key in names:
        # First try to get the value via getattr.  Some descriptors don't
        # like calling their __get__ (see bug #1785), so fall back to
        # looking in the __dict__.
        try:
            value = getattr(object, key)
            # handle the duplicate key
            if key in processed:
                raise AttributeError
        except AttributeError:
            for base in mro:
                if key in base.__dict__:
                    value = base.__dict__[key]
                    break
            else:
                # could be a (currently) missing slot member, or a buggy
                # __dir__; discard and move on
                continue
        if not predicate or predicate(value):
            results.append((key, value))
        processed.add(key)

    return results


def _parse_args_into_iterable(
    args: tuple[Any, ...] | tuple[Iterable[Any]],
) -> Iterable[Any]:
    if len(args) == 1 and isinstance(args[0], Iterable):
        return args[0]

    return args


def _all_equal(x: Iterable) -> bool:
    x = iter(x)
    try:
        first = next(x)
    except StopIteration:
        return True

    return all(first == x for x in x)
