import jax

import jax.numpy as jnp

jax.numpy.set_printoptions(precision=3, suppress=True)
from jax import core

from jax.extend import linear_util as lu
from functools import partial, update_wrapper

from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, tree_map
from jax.interpreters import ad, batching
from jax._src import ad_util

from jax.extend.core import Primitive
from jax._src.util import weakref_lru_cache, cache
from jax._src import util

from typing import Any, Callable
from jax._src.util import safe_map
from jax._src.api_util import (
    flatten_fun_nokwargs,
    argnums_partial,
    flatten_fun_nokwargs,
)
from jax._src.core import shaped_abstractify


from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe

# This is a custom primitive that allows us to define custom inverse functions
# While most stuff can be inverted by inverting all primitives for some functions it is necessary or more efficient to define a custom inverse function

custom_inverse_call_p = Primitive("custom_inverse_call_p")
custom_inverse_call_p.multiple_results = True


@custom_inverse_call_p.def_impl
def custom_inverse_call_impl(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        ans = core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.literals, *args)
    return ans


@custom_inverse_call_p.def_abstract_eval
def custom_inverse_call_abstract_eval(*args, forward_jaxpr, inverse_jaxpr, **params):
    with core.new_sublevel():
        return forward_jaxpr.out_avals


def custom_inverse_call_lowering(ctx, *args, forward_jaxpr, inverse_jaxpr, **params):
    return mlir.core_call_lowering(
        ctx, *args, name="forward_call", call_jaxpr=forward_jaxpr
    )


mlir.register_lowering(custom_inverse_call_p, custom_inverse_call_lowering)



def process_jvp(forward_jaxpr, tangents):
    nonzeros = [type(t) is not ad_util.Zero for t in tangents]
    forward_jvp_jaxpr, forward_out_nz = ad.jvp_jaxpr(
        forward_jaxpr, nonzeros, instantiate=False
    )
    nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
    # forward_jvp_jaxpr_ = pe.convert_constvars_jaxpr(forward_jvp_jaxpr.jaxpr)
    return forward_jvp_jaxpr, nonzero_tangents

#@custom_inverse_call_p.def_jvp
def custom_inverse_jvp(primals, tangents, forward_jaxpr, inverse_jaxpr, **params):
    forward_jvp_jaxpr, nonzero_tangents = process_jvp(forward_jaxpr, tangents)

    new_primals, new_tangent = core.eval_jaxpr(
        forward_jvp_jaxpr.jaxpr, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )

    return [
        new_primals,
    ], [
        new_tangent,
    ]



def _batch_custom_inverse_call_vmap(batched_args, batch_dims, **params):
    """Batching rule for `custom_inverse_call_p` for `vmap`."""

    # 1. Unpack the JAXPRs from the primitive's parameters.
    forward_jaxpr = params.pop("forward_jaxpr")
    inverse_jaxpr = params.pop("inverse_jaxpr")

    # 2. Batch the forward JAXPR.
    # The modern way to "batch a JAXPR" is to wrap its evaluation in a
    # Python function, apply `vmap` to that function, and then use
    # `make_jaxpr` to get the new, batched JAXPR.

    def fwd_eval_func(*args):
        # A helper function that evaluates the original forward jaxpr.
        return core.eval_jaxpr(forward_jaxpr.jaxpr, forward_jaxpr.literals, *args)

    # vmap this helper. `in_axes` are the batch dimensions of our inputs.
    # `out_axes=0` means the outputs will be batched on the first axis.
    vmapped_fwd_eval = vmap(fwd_eval_func, in_axes=batch_dims, out_axes=0)

    # Trace the vmapped function to get the new batched JAXPR.
    # We must provide the *unbatched* abstract values (avals) for the trace.
    unbatched_avals = [
        core.unmapped_aval(arg.aval, bdim) if bdim is not None else arg.aval
        for arg, bdim in zip(batched_args, batch_dims)
    ]
    new_forward_closed_jaxpr = make_jaxpr(vmapped_fwd_eval)(*unbatched_avals)


    # 3. Batch the inverse JAXPR using the same pattern.
    def inv_eval_func(*args):
        return core.eval_jaxpr(inverse_jaxpr.jaxpr, inverse_jaxpr.literals, *args)

    # The inputs to the inverse function are the outputs of the forward one.
    # Since we used `out_axes=0` above, the inputs here are batched on axis 0.
    vmapped_inv_eval = vmap(inv_eval_func, in_axes=0, out_axes=0)

    # Trace it using the output avals from the new forward jaxpr.
    new_inverse_closed_jaxpr = make_jaxpr(vmapped_inv_eval)(*new_forward_closed_jaxpr.out_avals)

    # 4. Recursively call the primitive.
    # We use the original batched arguments but pass the *newly created*
    # batched JAXPRs as parameters.
    out = custom_inverse_call_p.bind(
        *batched_args,
        forward_jaxpr=new_forward_closed_jaxpr,
        inverse_jaxpr=new_inverse_closed_jaxpr,
        **params
    )

    # 5. Specify the output batch dimensions.
    # Since we used `out_axes=0`, all outputs are batched on the first axis.
    out_dims = [0] * len(out)
    return out, out_dims

#@custom_inverse_call_p.def_transpose
def custom_inverse_transpose(*args, **kwargs):
    return ad.call_transpose(custom_inverse_call_p, *args, **kwargs)

#batching.spmd_axis_primitive_batchers[custom_inverse_call_p] = batch_custom_inverse_call
batching.axis_primitive_batchers[custom_inverse_call_p] = partial(
    _batch_custom_inverse_call_vmap, None
)
ad.primitive_transposes[custom_inverse_call_p] = custom_inverse_transpose
ad.primitive_jvps[custom_inverse_call_p] = custom_inverse_jvp


def is_hashable(obj):
    try:
        hash(obj)
        return True
    except TypeError:
        return False


# TODO: Add support other tracer support!


@jax._src.util.cache()
def trace_forward_inverse(
    f,
    f_inv,
    dyn_args_index,
    inv_argnum,
    in_avals,
    in_tree,
    name,
):
    # print(in_avals)
    # Forward
    f, out_tree = flatten_fun_nokwargs(f, in_tree)  # type: ignore
    debug = pe.debug_info(f.f, in_tree, out_tree, False, name or "<unknown>")
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(f, in_avals, debug)
    forward_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    out_tree = out_tree()

    # Inverse
    f_inv, _ = flatten_fun_nokwargs(f_inv, in_tree)  # type: ignore
    inv_in_avals = list(in_avals)
    i = dyn_args_index.index(inv_argnum)
    inv_in_avals[i] = out_avals[0]
    # print(inv_in_avals)

    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(f_inv, inv_in_avals, debug)
    inverse_jaxpr = core.ClosedJaxpr(jaxpr, consts)

    return forward_jaxpr, inverse_jaxpr, out_tree


class custom_inverse:
    def __init__(self, fun: Callable, inv_argnum=0, static_argnums=None) -> None:
        update_wrapper(self, fun)
        self.fun = fun
        self.static_argnums = static_argnums
        self.inv_argnum = inv_argnum

    def definv(self, inv_fun: Callable) -> Callable:
        def wrapped_inv(*args, **kwargs):
            return inv_fun(*args, **kwargs), jnp.nan

        self.inv_fun = inv_fun
        self.inv_fun_and_log_det = wrapped_inv
        return wrapped_inv

    def definv_and_logdet(self, inv_fun_and_log_det: Callable) -> Callable:
        self.inv_fun_and_log_det = inv_fun_and_log_det
        if not hasattr(self, "inv_fun"):
            self.inv_fun = lambda *args, **kwargs: inv_fun_and_log_det(*args, **kwargs)[
                0
            ]
        return inv_fun_and_log_det

    def inv(self, *args, **kwargs):
        return self.inv_fun(*args, **kwargs)

    def inv_and_logdet(self, *args, **kwargs):
        return self.inv_fun_and_log_det(*args, **kwargs)

    def __call__(self, *args, **params) -> Any:
        name = getattr(self.fun, "__name__", str(self.fun))
        if not self.inv_fun:
            msg = f"No inverse defined for custom_inverse function {name} using definv."
            raise AttributeError(msg)
        inv_name = getattr(self.inv_fun, "__name__", str(self.inv_fun))

        # We can only invert with respect to specific dynamic arguments. All others are assumed to be static!
        f = lu.wrap_init(self.fun, params=params)
        f_inv = lu.wrap_init(self.inv_fun_and_log_det, params=params)

        # Dynamic and static args for forward and inverse
        if self.static_argnums is None:
            dyn_args = args
            dyn_args_index = tuple(i for i in range(len(args)))
        else:
            dyn_args_index = tuple(
                [
                    i
                    for i in range(len(args))
                    if i not in self.static_argnums  # or not is_hashable(args[i])
                ]
            )

            f, dyn_args = argnums_partial(
                f, dyn_args_index, args, require_static_args_hashable=True
            )

            f_inv, _ = argnums_partial(
                f_inv, dyn_args_index, args, require_static_args_hashable=True
            )

        # print(dyn_args, args, self.static_argnums)
        # Flatt stuff for tracing
        args_flat, in_tree = tree_flatten(dyn_args)
        in_avals = tuple(safe_map(shaped_abstractify, args_flat))


        forward_jaxpr, inverse_jaxpr, out_tree = trace_forward_inverse(
            f,
            f_inv,
            dyn_args_index,
            self.inv_argnum,
            in_avals,
            in_tree,
            name,
        )

        out_flat = custom_inverse_call_p.bind(
            *args_flat,
            forward_jaxpr=forward_jaxpr,
            inverse_jaxpr=inverse_jaxpr,
            in_tree=in_tree,
            inv_argnum=dyn_args_index.index(self.inv_argnum),
        )

        return tree_unflatten(out_tree, out_flat)
