import sys

import numpy as np
import theano
from theano import tensor
import pyopencl as cl

def create_vm(updated_vars, vals_memo,
        allow_gc=True, use_cloop=True, callback=None):
    """
    updated_vars: sequence of (dst, expr) pairs

    allow_gc - force the virtual machine to clean up unnecessary references,
        in order to allow garbage collection on intermediate values during
        computation of a function.

    use_cloop - use the C-based virtual machine if possible

    callback - a callable object to call after each call to a thunk within
        the virtual machine.  It will be called with four arguments called
        'node', 'thunk', 'storage_map', and 'compute_map'.

    """
    gof = theano.gof
    link = gof.link
    Loop = gof.vm.Loop
    LoopGC = gof.vm.LoopGC
    Stack = gof.vm.Stack
    CVM = gof.vm.CVM


    outputs = [expr for dst, expr in updated_vars]
    dests = [dst for dst, expr in updated_vars]
    inputs = gof.graph.inputs(outputs + dests)
    env = gof.Env(inputs, outputs)
    no_recycling = []
    order = env.toposort()

    input_storage = [vals_memo[i] if i in vals_memo else [i.data]
            for i in inputs]
    output_storage = [vals_memo[i] if i in vals_memo else [None]
            for i in outputs]

    input_storage, output_storage, storage_map = gof.link.map_storage(
            env, order, input_storage, output_storage)

    compute_map = {}
    for k in storage_map:
        compute_map[k] = [k.owner is None]

    thunks = map(
            lambda node: node.op.make_thunk(
                node, storage_map, compute_map, no_recycling),
            order)

    computed, last_user = link.gc_helper(order)
    if allow_gc:
        post_thunk_clear = []
        for node in order:
            clear_after_this_thunk = []
            for input in node.inputs:
                if ((input in computed)
                        and (input not in env.outputs)
                        and (node == last_user[input])):
                    clear_after_this_thunk.append(storage_map[input])
            post_thunk_clear.append(clear_after_this_thunk)
    else:
        post_thunk_clear = None

    pre_call_clear = [storage_map[v] for v in no_recycling]

    if callback is not None:
        if use_cloop:
            logger.warn('CLoop does not support callback, using Stack VM.')
        vm = Stack(
                order, thunks, pre_call_clear,
                storage_map, compute_map,
                env, allow_gc,
                callback=callback)
    elif use_cloop:
        # create a map from nodes to ints and vars to ints
        nodes_idx = {}
        vars_idx = {}
        for i, node in enumerate(order):
            nodes_idx[node] = i
            for v in node.inputs + node.outputs:
                vars_idx.setdefault(v, len(vars_idx))
        for v in env.inputs + env.outputs:
            vars_idx.setdefault(v, len(vars_idx))

        nodes_idx_inv = {}
        vars_idx_inv = {}
        for (node,i) in nodes_idx.items():
            nodes_idx_inv[i] = node
        for (var,i) in vars_idx.items():
            vars_idx_inv[i] = var

        # put storage_map and compute_map into a int-based scheme
        n_applies = len(order)
        storage_map_list = [storage_map[vars_idx_inv[i]]
                for i in xrange(len(vars_idx_inv))]
        compute_map_list = [compute_map[vars_idx_inv[i]]
                for i in xrange(len(vars_idx_inv))]
        if order:
            assert type(storage_map_list[0]) is list
            assert type(compute_map_list[0]) is list

        # build the pointers to node inputs and offsets
        base_input_output_list = []
        node_n_inputs = []
        node_n_outputs = []
        node_input_offset = []
        node_output_offset = []
        for node in order:
            inputs_idx = [vars_idx[v] for v in node.inputs]
            outputs_idx = [vars_idx[v] for v in node.outputs]
            node_n_inputs.append(len(inputs_idx))
            node_n_outputs.append(len(outputs_idx))
            node_input_offset.append(len(base_input_output_list))
            base_input_output_list.extend(inputs_idx)
            node_output_offset.append(len(base_input_output_list))
            base_input_output_list.extend(outputs_idx)

        # build the var owner array
        var_owner = [None]*len(vars_idx)
        for (var,i) in vars_idx.items():
            if var.owner:
                var_owner[i] = nodes_idx[var.owner]

        is_lazy_list = [int(th.lazy) for th in thunks]
        output_vars = [vars_idx[v] for v in env.outputs]

        # builds the list of prereqs induced by e.g. destroy_handler
        ords = env.orderings()
        node_prereqs = []
        node_output_size = []
        for i, node in enumerate(order):
            node_output_size.append(0)
            prereq_var_idxs = []
            for prereq_node in ords.get(node,[]):
                prereq_var_idxs.extend(
                        [vars_idx[v] for v in prereq_node.outputs])
            prereq_var_idxs = list(set(prereq_var_idxs))
            prereq_var_idxs.sort() # TODO: why sort?
            node_prereqs.append(prereq_var_idxs)

        update_storage = []
        for (ivar, ovar) in updated_vars:
            if ivar != ovar:
                update_storage.append(vars_idx[ivar]) #dst
                update_storage.append(vars_idx[ovar]) #src

        c0 = sys.getrefcount(node_n_inputs)
        vm = CVM(
                order,
                thunks,
                pre_call_clear,
                allow_gc=allow_gc,
                call_counts=[0]*len(order),
                call_times=[0.0]*len(order),
                compute_map_list=compute_map_list,
                storage_map_list=storage_map_list,
                base_input_output_list=base_input_output_list,
                node_n_inputs=node_n_inputs,
                node_n_outputs=node_n_outputs,
                node_input_offset=node_input_offset,
                node_output_offset=node_output_offset,
                var_owner=var_owner,
                is_lazy_list=is_lazy_list,
                output_vars=output_vars,
                node_prereqs=node_prereqs,
                node_output_size=node_output_size,
                update_storage=update_storage,
                )
        assert c0 == sys.getrefcount(node_n_inputs)
    else:
        if all([(not th.lazy) for th in thunks]):
            # there is no conditional in the graph
            if allow_gc:
                vm = LoopGC(
                        order,
                        thunks,
                        pre_call_clear,
                        post_thunk_clear)
            else:
                vm = Loop(
                        order,
                        thunks,
                        pre_call_clear)
        else:
            vm = Stack(
                    order, thunks, pre_call_clear,
                    storage_map, compute_map,
                    env, allow_gc
                    )

    return vm


class Workspace(object):
    """
    """

    def __init__(self ):
        self.ctx = cl.Context(dev_type=cl.device_type.CPU)
        self.queue = cl.CommandQueue(self.ctx)
        self.vals_memo = {}

    def __contains__(self, key):
        return key in self.vals_memo

    def __getitem__(self, key):
        return self.vals_memo[key][0]
        #cl.enqueue_read_buffer(self.queue, z_buf, z).wait()

    def __setitem__(self, key, val):
        self.vals_memo[key] = [val]

    def update(self, dct):
        raise NotImplentedError()

    def _computed_update(self, updated_vars):
        """

        Return a function that will update this workspaces
        values for keys according to `exprs`

        """
        return create_vm(updated_vars, self.vals_memo)

    def computed_update(self, dct):
        """
        {x: 2 * y, z: 3 * x + 4}
        """
        fn = self._computed_update(dct)
        fn()
        return fn


if __name__ == '__main__':
    ws = Workspace()

    x = tensor.vector()
    y = tensor.vector()
    ws[x] = np.random.randn(2)
    ws[y] = np.random.randn(2)

    f = ws._computed_update([
        (x, 2 * x),
        (y, x + y)])
    
    for i in range(6):
        f()
        print ws[x], ws[y]
