__all__ = ["describe"]

import logging
import random
import string

import scipy as sp

# need to check scipy version since the interface submodule changed into
# _interface from scipy>=1.8.0
sp_version = sp.__version__.split(".")
if int(sp_version[0]) <= 1 and int(sp_version[1]) < 8:
    from scipy.sparse.linalg.interface import (
        _AdjointLinearOperator,
        _ProductLinearOperator,
        _TransposedLinearOperator,
    )
else:
    from scipy.sparse.linalg._interface import (
        _AdjointLinearOperator,
        _ProductLinearOperator,
        _TransposedLinearOperator,
    )

from typing import List, Set, Union

from pylops import LinearOperator
from pylops.basicoperators import BlockDiag, HStack, VStack
from pylops.linearoperator import _ScaledLinearOperator, _SumLinearOperator
from pylops.utils import deps

sympy_message = deps.sympy_import("the describe module")

if sympy_message is None:
    from sympy import BlockDiagMatrix, BlockMatrix, MatrixSymbol


compositeops = (
    LinearOperator,
    _SumLinearOperator,
    _ProductLinearOperator,
    _ScaledLinearOperator,
    _AdjointLinearOperator,
    _TransposedLinearOperator,
    HStack,
    VStack,
    BlockDiag,
)

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)


def _in_notebook() -> bool:
    """Check if code is running inside notebook"""
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def _assign_name(Op, Ops, names: List[str]) -> str:
    """Assign name to an operator as provided by the user
    (or randomly select one when not provided by the user)

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to assign name to
    Ops : :obj:`dict`
        dictionary of Operators found by the _describe method whilst crawling
        through the composite operator to describe
    names : :obj:`list`
        list of currently assigned names

    Returns
    -------
    name : :obj:`str`
        Name assigned to operator

    """
    # Add a suffix when all letters of the alphabet are already in use. This
    # decision is made by counting the length of the names list and using the
    # English vocabulary (26 characters)
    suffix = str(len(names) // 26) * (len(names) // 26 > 0)

    # Propose a new name, where a random letter
    # is chosen if Op does not have a name or the name is set to None
    if getattr(Op, "name", None) is None:
        proposedname = random.choice(string.ascii_letters).upper() + suffix
    else:
        proposedname = Op.name + suffix

    if proposedname not in names or (Ops[proposedname][1] == id(Op)):
        # Assign the proposed name if this is not yet in use or if it is
        # used by the same operator. Note that an operator may reapper
        # multiple times in an expression
        name = proposedname
    else:
        # Propose a new name until an unused character is found
        origname = proposedname
        while proposedname in names:
            proposedname = random.choice(string.ascii_letters).upper() + suffix
        name = proposedname
        logging.warning(
            f"The user has used the same name {origname} for two distinct operators, "
            f"changing name of operator {type(Op).__name__} to {name}..."
        )
    Op.name = name
    return name


def _describeop(Op, Ops, names: List[str]):
    """Core steps to describe a single operator

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to assign name to
    Ops : :obj:`dict`
        dictionary of Operators found by the _describe method whilst crawling
        through the composite operator to describe
    names : :obj:`list`
        list of currently assigned names

    Returns
    -------
    Op0 : :obj:`sympy.MatrixSymbol`
        Sympy equivalent od Linear Operator ``Op``
    Ops_ : :obj:`dict`
        New or updated dictionary of Operators

    """
    if type(Op) not in compositeops:
        # A native PyLops operator has been found, assign a name and store
        # it as MatrixSymbol
        name = _assign_name(Op, Ops, names)
        Op0 = MatrixSymbol(name, 1, 1)
        Ops_ = {name: (type(Op).__name__, id(Op))}
    elif type(Op) is LinearOperator:
        # A LinearOperator has been found, either extract Op and start to
        # describe it or if a name has been given to the operator treat as
        # it is (this is useful when users do not want an operator to be
        # further disected into its components
        name = getattr(Op, "name", None)
        if name is None:
            Op0, Ops_, names = _describe(Op.Op, Ops, names)
        else:
            Ops_ = {name: (type(Op).__name__, id(Op))}
            Op0 = MatrixSymbol(name, 1, 1)
    else:
        # When finding a composite operator, send it again to the _describe
        # method
        Op0, Ops_, names = _describe(Op, Ops, names)
    return Op0, Ops_


def _describe(
    Op,
    Ops,
    names: Union[List[str], Set[str]],
):
    """Core steps to describe a composite operator. This is done recursively.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to assign name to
    Ops : :obj:`dict`
        dictionary of Operators found by the _describe method whilst crawling
        through the composite operator to describe
    names : :obj:`list`
        list of currently assigned names

    Returns
    -------
    Opsym : :obj:`sympy.MatrixSymbol`
        Sympy equivalent od Linear Operator ``Op``
    Ops : :obj:`dict`
        dictionary of Operators

    """
    # Check if a name has been given to the operator and store it as
    # MatrixSymbol (this is useful when users do not want an operator to be
    # further disected into its components)
    name = getattr(Op, "name", None)
    if name is not None:
        Ops[name] = (type(Op).__name__, id(Op))
        Opsym = MatrixSymbol(Op.name, 1, 1)
        names.update(name)
        return Opsym, Ops, names

    # Given that no name has been assigned, interpret the operator further
    if type(Op) not in compositeops:
        # A native PyLops operator has been found, assign a name
        # or if a name has been given to the operator treat as
        # it is and store it as MatrixSymbol
        name = _assign_name(Op, Ops, list(Ops.keys()))
        Ops[name] = (type(Op).__name__, id(Op))
        Opsym = MatrixSymbol(Op.name, 1, 1)
        names.update(name)
    else:
        if type(Op) is LinearOperator:
            # A LinearOperator has been found, either extract Op and start to
            # describe it or if a name has been given to the operator treat as
            # it is and store it as MatrixSymbol
            Opsym, Ops_, names = _describe(Op.Op, Ops, names)
            Ops.update(Ops_)
        elif type(Op) is _AdjointLinearOperator:
            # An adjoint LinearOperator has been found, describe it and attach
            # the adjoint symbol to its sympy representation
            Opsym, Ops_, names = _describe(Op.args[0], Ops, names)
            Opsym = Opsym.adjoint()
            Ops.update(Ops_)
        elif type(Op) is _TransposedLinearOperator:
            # A transposed LinearOperator has been found, describe it and
            # attach the transposed symbol to its sympy representation
            Opsym, Ops_, names = _describe(Op.args[0], Ops, names)
            Opsym = Opsym.T
            Ops.update(Ops_)
        elif type(Op) is _ScaledLinearOperator:
            # A scaled LinearOperator has been found, describe it and
            # attach the scaling to its sympy representation. Note that the
            # scaling could either on the left or right side of the operator,
            # so we need to try both
            if isinstance(Op.args[0], LinearOperator):
                Opsym, Ops_ = _describeop(Op.args[0], Ops, names)
                Opsym = Op.args[1] * Opsym
                Ops.update(Ops_)
                names.update(list(Ops_.keys()))
            else:
                Opsym, Ops_ = _describeop(Op.args[1], Ops, names)
                Opsym = Op.args[1] * Opsym
                Ops.update(Ops_)
                names.update(list(Ops_.keys()))
        elif type(Op) is _SumLinearOperator:
            # A sum LinearOperator has been found, describe both operators
            # either side of the plus sign and sum their sympy representations
            Opsym0, Ops_ = _describeop(Op.args[0], Ops, names)
            Ops.update(Ops_)
            names.update(list(Ops_.keys()))
            Opsym1, Ops_ = _describeop(Op.args[1], Ops, names)
            Ops.update(Ops_)
            names.update(list(Ops_.keys()))
            Opsym = Opsym0 + Opsym1
        elif type(Op) is _ProductLinearOperator:
            # Same as sum LinearOperator but for product
            Opsym0, Ops_ = _describeop(Op.args[0], Ops, names)
            Ops.update(Ops_)
            names.update(list(Ops_.keys()))
            Opsym1, Ops_ = _describeop(Op.args[1], Ops, names)
            Ops.update(Ops_)
            names.update(list(Ops_.keys()))
            Opsym = Opsym0 * Opsym1
        elif type(Op) in (VStack, HStack, BlockDiag):
            # A special composite operator has been found, stack its components
            # horizontally, vertically, or along a diagonal
            Opsyms = []
            for op in Op.ops:
                Opsym, Ops_ = _describeop(op, Ops, names)
                Opsyms.append(Opsym)
                names.update(list(Ops_.keys()))
                Ops.update(Ops_)
            Ops.update(Ops_)
            if type(Op) is VStack:
                Opsym = BlockMatrix([[Opsym] for Opsym in Opsyms])
            elif type(Op) is HStack:
                Opsym = BlockMatrix(Opsyms)
            elif type(Op) is BlockDiag:
                Opsym = BlockDiagMatrix(*Opsyms)
    return Opsym, Ops, names


def describe(Op) -> None:
    r"""Describe a PyLops operator

    .. versionadded:: 1.17.0

    Convert a PyLops operator into a ``sympy`` mathematical formula.
    This routine is useful both for debugging and educational purposes.

    Note that users can add a name to each operator prior to
    running the describe method, i.e. ``Op.name='Opname'``. Alternatively, each
    of the PyLops operator that composes the operator ``Op`` is automatically
    assigned a name. Moreover, note that the symbols :math:`T` and
    :math:`\dagger` are used in the mathematical expressions to indicate
    transposed and adjoint operators, respectively.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to describe

    """
    if sympy_message is not None:
        raise NotImplementedError(sympy_message)

    # Describe the operator
    Ops = {}
    names = set()
    Opsym, Ops, names = _describe(Op, Ops=Ops, names=names)
    # Clean up Ops from id
    Ops = {op: Ops[op][0] for op in Ops.keys()}
    # Check if this command is run in a Jupyter notebook or normal shell and
    # display the operator accordingly
    if _in_notebook():
        from IPython.display import display

        display(Opsym)
    else:
        print(Opsym)
    print("where:", Ops)
