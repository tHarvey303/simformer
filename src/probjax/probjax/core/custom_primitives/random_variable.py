from jax.extend.core import (
    Primitive,
    ClosedJaxpr,
)
from jax.core import eval_jaxpr, ShapedArray

import jax
import jax.random as jrandom
from jax import tree_util
from jax.extend import linear_util as lu
from jax._src import api_util
from jax._src import ad_util
from jax._src import util
from typing import Hashable, Callable
from jax._src import effects
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import mlir
from jax.interpreters.batching import batch_jaxpr
from jax.interpreters import partial_eval as pe
from jax._src.core import shaped_abstractify

from jax._src.util import safe_map as map

from functools import partial


from probjax.distributions.distribution import Distribution

__all__ = ["rv", "rv_p"]


def _sample_distribution(dist: Distribution, key, *args, shape=(), **kwargs):
    return dist.sample(key, *args, sample_shape=shape, **kwargs)


def _log_prob_distribution(dist: Distribution, value, *args, **kwargs):
    return dist.log_prob(value=value, *args, **kwargs)


# This maybe should be refactored
@util.cache()
def _sampling_logprobs_jaxprs_with_common_consts(sampling_fn, log_prob_fn):
    wrapped_sampling_fn = lu.wrap_init(sampling_fn)
    in_avals = [
        ShapedArray((2,), jax.numpy.uint32),
    ]  # The PRNG Key!
    in_tree = tree_util.tree_structure(in_avals)
    flat_wrapped_sampling_fn, out_tree = api_util.flatten_fun_nokwargs(  # type: ignore
        wrapped_sampling_fn, in_tree
    )
    debug = pe.debug_info(sampling_fn, in_tree, out_tree, False, "sampling_fn")
    sampling_jaxpr, sampling_out_avals, sampling_consts = pe.trace_to_jaxpr_dynamic(
        flat_wrapped_sampling_fn, in_avals, debug
    )

    wrapped_log_prob_fn = lu.wrap_init(log_prob_fn)
    log_prob_operands = sampling_out_avals
    flat_log_prob_operands, log_prob_in_tree = tree_util.tree_flatten(log_prob_operands)
    flat_wrapped_log_prob_fn, log_prob_out_tree = api_util.flatten_fun_nokwargs(  # type: ignore
        wrapped_log_prob_fn, log_prob_in_tree
    )
    debug = pe.debug_info(
        log_prob_fn, log_prob_in_tree, log_prob_out_tree, False, "log_prob_fn"
    )
    log_prob_jaxpr, log_prob_out_avals, log_prob_consts = pe.trace_to_jaxpr_dynamic(
        flat_wrapped_log_prob_fn, flat_log_prob_operands, debug
    )

    jaxprs = [sampling_jaxpr, log_prob_jaxpr]
    consts = [sampling_consts, log_prob_consts]
    # out_trees = [sampling_out_trees, log_prob_out_trees]

    newvar = jax._src.core.gensym(jaxprs, suffix="_")  # type: ignore
    all_const_avals = [map(shaped_abstractify, consts) for consts in consts]
    unused_const_vars = [map(newvar, const_avals) for const_avals in all_const_avals]

    def pad_jaxpr_constvars(i, jaxpr):
        prefix = util.concatenate(unused_const_vars[:i])
        suffix = util.concatenate(unused_const_vars[i + 1 :])
        constvars = [*prefix, *jaxpr.constvars, *suffix]
        return jaxpr.replace(constvars=constvars)

    consts = util.concatenate(consts)
    jaxprs = tuple(pad_jaxpr_constvars(i, jaxpr) for i, jaxpr in enumerate(jaxprs))
    closed_jaxprs = [
        ClosedJaxpr(pe.convert_constvars_jaxpr(jaxpr), ()) for jaxpr in jaxprs
    ]

    return closed_jaxprs, consts


def rv(dist: Distribution, name: Hashable) -> Callable:
    """This takes a distribution and returns a function that samples from that distribution.


    Args:
        dist (Distribution): Distribution of random variable
        name (Hashable): Name of random variable

    Returns:
        Callable: Sampling function
    """

    def sample_fn(key, *args, **kwargs):
        return _sample_distribution(dist, key, *args, **kwargs)

    def log_prob_fn(value, *args, **kwargs):
        return _log_prob_distribution(dist, value, *args, **kwargs)

    (
        [sampling_fn_jaxpr, log_prob_fn_jaxpr],
        consts,
    ) = _sampling_logprobs_jaxprs_with_common_consts(sample_fn, log_prob_fn)

    def wrapped(*args, **kwargs):
        out = rv_p.bind(
            *consts,
            *args,
            name=name,
            sampling_fn_jaxpr=sampling_fn_jaxpr,
            log_prob_fn_jaxpr=log_prob_fn_jaxpr,
            dist=type(dist),
            intervened=False,
            **kwargs
        )

        return out[0]

    return wrapped

#@rv_p.def_impl
def _rv_impl(*args, **params):

    call_jaxpr = params["sampling_fn_jaxpr"]
    return eval_jaxpr(call_jaxpr.jaxpr, call_jaxpr.literals, *args)

#@rv_p.def_abstract_eval
def _rv_abstract_eval(*args, **params):
    call_jaxpr = params["sampling_fn_jaxpr"]
    return call_jaxpr.out_avals


# JIT support
def _rv_lowering(ctx, *args, name, sampling_fn_jaxpr, log_prob_fn_jaxpr, **params):
    call_jaxpr = sampling_fn_jaxpr
    return mlir.core_call_lowering(ctx, *args, name=name, call_jaxpr=call_jaxpr)


rv_p = Primitive("random_variable")
rv_p.multiple_results = True

def _rv_transpose_rule(*args, **kwargs):
    return ad.call_transpose(rv_p, *args, **kwargs)

def _rv_batching_rule(batched_args, batch_dims, **params):
    """Modern batching rule for the 'random_variable' primitive."""

    # 1. Unpack the JAXPRs from the primitive's parameters.
    sampling_fn_jaxpr = params.pop("sampling_fn_jaxpr")
    log_prob_fn_jaxpr = params.pop("log_prob_fn_jaxpr")

    # 2. Batch the sampling JAXPR using the vmap -> make_jaxpr pattern.
    def sampling_eval_func(*args):
        # Helper to evaluate the original, unbatched sampling jaxpr.
        return core.eval_jaxpr(sampling_fn_jaxpr.jaxpr, sampling_fn_jaxpr.literals, *args)

    # vmap the helper function. `in_axes` are the batch dimensions of our inputs.
    vmapped_sampling_eval = vmap(sampling_eval_func, in_axes=batch_dims, out_axes=0)

    # Trace the vmapped function to get the new batched JAXPR.
    # We trace with the *unbatched* abstract values (avals).
    unbatched_avals = [
        core.unmapped_aval(arg.aval, bdim) if bdim is not None else arg.aval
        for arg, bdim in zip(batched_args, batch_dims)
    ]
    new_sampling_closed_jaxpr = make_jaxpr(vmapped_sampling_eval)(*unbatched_avals)

    # 3. Batch the log_prob JAXPR using the same pattern.
    def log_prob_eval_func(*args):
        # Helper to evaluate the original log_prob jaxpr.
        return core.eval_jaxpr(log_prob_fn_jaxpr.jaxpr, log_prob_fn_jaxpr.literals, *args)

    # The inputs to the log_prob function are the outputs of the sampling one.
    # Since we used `out_axes=0` for the sampling vmap, the inputs here are
    # batched on axis 0.
    vmapped_log_prob_eval = vmap(log_prob_eval_func, in_axes=0, out_axes=0)

    # Trace it using the output avals from the new sampling jaxpr.
    new_log_prob_closed_jaxpr = make_jaxpr(vmapped_log_prob_eval)(*new_sampling_closed_jaxpr.out_avals)

    # 4. Recursively call the primitive with the new batched JAXPRs.
    out = rv_p.bind(
        *batched_args,
        sampling_fn_jaxpr=new_sampling_closed_jaxpr,
        log_prob_fn_jaxpr=new_log_prob_closed_jaxpr,
        **params
    )

    # 5. Specify the output batch dimensions.
    # Since we vmapped with `out_axes=0`, all outputs are batched on the first axis.
    out_dims = [0] * len(out)
    return out, out_dims


#@rv_p.def_jvp
def custom_inverse_jvp(primals, tangents, sampling_fn_jaxpr, **params):
    nonzeros =  [type(t) is not ad_util.Zero for t in tangents]
    forward_jvp_jaxpr, forward_out_nz = ad.jvp_jaxpr(
        sampling_fn_jaxpr, nonzeros, instantiate=False
    )
    nonzero_tangents = [t for t in tangents if type(t) is not ad_util.Zero]
    forward_jvp_jaxpr_ = pe.convert_constvars_jaxpr(forward_jvp_jaxpr.jaxpr)

    new_primals, new_tangent = eval_jaxpr(
        forward_jvp_jaxpr_, forward_jvp_jaxpr.consts, *primals, *nonzero_tangents
    )

    return new_primals, new_tangent


rv_p.def_impl(_rv_impl)
rv_p.def_abstract_eval(_rv_abstract_eval)


#batching.spmd_axis_primitive_batchers[rv_p] = _rv_batching_rule
batching.axis_primitive_batchers[rv_p] = partial(_rv_batching_rule, None)
mlir.register_lowering(rv_p, _rv_lowering)
ad.primitive_transposes[rv_p] = _rv_transpose_rule
ad.primitive_jvps[rv_p] = custom_inverse_jvp
