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
        _SumLinearOperator,
        _TransposedLinearOperator,
    )
else:
    from scipy.sparse.linalg._interface import (
        _AdjointLinearOperator,
        _ProductLinearOperator,
        _SumLinearOperator,
        _TransposedLinearOperator,
    )

from pylops import LinearOperator
from pylops.basicoperators import BlockDiag, HStack, VStack
from pylops.LinearOperator import _ScaledLinearOperator

try:
    from sympy import BlockDiagMatrix, BlockMatrix, MatrixSymbol
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Sympy package not installed. In order to use "
        "the describe method run "
        "install sympy."
    )

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


def _in_notebook():
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


def _assign_name(Op, Ops, names):
    """Assign name to an operator as provided by the user
    (or randomly select one when not provided by the user)

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to assign name to
    Ops : :obj:`dict`
        Dictionary of Operators found by the _describe method whilst crawling
        through the composite operator to describe
    names : :obj:`list`
        List of currently assigned names

    """
    # Add a suffix when all letters of the alphabet are already in use. This
    # decision is made by counting the length of the names list and using the
    # English vocabulary (26 characters)
    suffix = "1" * (len(names) // 26)
    # Propose a new name, where a random letter is chosen if Op does not
    # have a name
    proposedname = (
        getattr(Op, "name", random.choice(string.ascii_letters).upper()) + suffix
    )
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
        print(
            f"The user has used the same name {origname} for two distinct operators, "
            f"changing name of operator {type(Op).__name__} to {name}..."
        )
    Op.name = name
    return name


def _describeop(Op, Ops, names):
    """Core steps to describe a single operator

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to assign name to
    Ops : :obj:`dict`
        Dictionary of Operators found by the _describe method whilst crawling
        through the composite operator to describe
    names : :obj:`list`
        List of currently assigned names

    """
    if type(Op) not in compositeops:
        # A native PyLops operator has been found, assign a name and store
        # it as MatrixSymbol
        name = _assign_name(Op, Ops, names)
        Op0 = MatrixSymbol(name, 1, 1)
        Ops_ = {name: (type(Op).__name__, id(Op))}
    elif type(Op) == LinearOperator:
        # A LinearOperator has been found, either extract Op and start to
        # describe it or if a name has been given to the operator treat as
        # it is (this is useful when users do not want an operator to be
        # further disected into its components
        name = getattr(Op, "name", None)
        if name is None:
            Op0, Ops_ = _describe(Op.Op)
        else:
            Ops_ = {name: (type(Op).__name__, id(Op))}
            Op0 = MatrixSymbol(name, 1, 1)
    else:
        # When finding a composite operator, send it again to the _describe
        # method
        Op0, Ops_ = _describe(Op)
    return Op0, Ops_


def _describe(Op):
    """Core steps to describe a composite operator. This is done recursively.

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear Operator to assign name to
    Ops : :obj:`dict`
        Dictionary of Operators found by the _describe method whilst crawling
        through the composite operator to describe
    names : :obj:`list`
        List of currently assigned names

    """
    Ops = {}
    if type(Op) not in compositeops:
        # A native PyLops operator has been found, assign a name and store
        # it as MatrixSymbol
        name = _assign_name(Op, Ops, list(Ops.keys()))
        Ops[name] = (type(Op).__name__, id(Op))
        Opsym = MatrixSymbol(Op.name, 1, 1)
    else:
        if type(Op) == LinearOperator:
            # A LinearOperator has been found, either extract Op and start to
            # describe it or if a name has been given to the operator treat as
            # it is (this is useful when users do not want an operator to be
            # further disected into its components
            name = getattr(Op, "name", None)
            if name is None:
                Opsym, Ops_ = _describe(Op.Op)
                Ops.update(Ops_)
            else:
                Ops[name] = (type(Op).__name__, id(Op))
                Opsym = MatrixSymbol(Op.name, 1, 1)
        elif type(Op) == _AdjointLinearOperator:
            # An adjoint LinearOperator has been found, describe it and attach
            # the adjoint symbol to its sympy representation
            Opsym, Ops_ = _describe(Op.args[0])
            Opsym = Opsym.adjoint()
            Ops.update(Ops_)
        elif type(Op) == _TransposedLinearOperator:
            # A transposed LinearOperator has been found, describe it and
            # attach the transposed symbol to its sympy representation
            Opsym, Ops_ = _describe(Op.args[0])
            Opsym = Opsym.T
            Ops.update(Ops_)
        elif type(Op) == _ScaledLinearOperator:
            # A scaled LinearOperator has been found, describe it and
            # attach the scaling to its sympy representation. Note that the
            # scaling could either on the left or right side of the operator,
            # so we need to try both
            if isinstance(Op.args[0], LinearOperator):
                Opsym, Ops_ = _describeop(Op.args[0], Ops, list(Ops.keys()))
                Opsym = Op.args[1] * Opsym
                Ops.update(Ops_)
            else:
                Opsym, Ops_ = _describeop(Op.args[1], Ops, list(Ops.keys()))
                Opsym = Op.args[1] * Opsym
                Ops.update(Ops_)
        elif type(Op) == _SumLinearOperator:
            # A sum LinearOperator has been found, describe both operators
            # either side of the plus sign and sum their sympy representations
            Opsym0, Ops_ = _describeop(Op.args[0], Ops, list(Ops.keys()))
            Ops.update(Ops_)
            Opsym1, Ops_ = _describeop(Op.args[1], Ops, list(Ops.keys()))
            Ops.update(Ops_)
            Opsym = Opsym0 + Opsym1
        elif type(Op) == _ProductLinearOperator:
            # Same as sum LinearOperator but for product
            Opsym0, Ops_ = _describeop(Op.args[0], Ops, list(Ops.keys()))
            Ops.update(Ops_)
            Opsym1, Ops_ = _describeop(Op.args[1], Ops, list(Ops.keys()))
            Ops.update(Ops_)
            Opsym = Opsym0 * Opsym1
        elif type(Op) in (VStack, HStack, BlockDiag):
            # A special composite operator has been found, stack its components
            # horizontally, vertically, or along a diagonal
            Opsyms = []
            for op in Op.ops:
                Opsym, Ops_ = _describeop(op, Ops, list(Ops.keys()))
                Opsyms.append(Opsym)
                Ops.update(Ops_)
            Ops.update(Ops_)
            if type(Op) == VStack:
                Opsym = BlockMatrix([[Opsym] for Opsym in Opsyms])
            elif type(Op) == HStack:
                Opsym = BlockMatrix(Opsyms)
            elif type(Op) == BlockDiag:
                Opsym = BlockDiagMatrix(*Opsyms)
    return Opsym, Ops


def describe(Op):
    """Describe a PyLops operator

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
    # Describe the operator
    Opsym, Ops = _describe(Op)
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
