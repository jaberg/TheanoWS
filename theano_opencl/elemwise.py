
import numpy, theano
from theano import tensor as T

import pyopencl as cl
class PowCL(theano.gof.Op):
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))
    def make_node(self, a, b):
        return theano.Apply(self,[a,b], [a.type()])

    def make_thunk(self, node,
            input_computed, output_computed,
            input_registers, output_registers):

        def rval():
            a,b = input_registers
            output_registers[0] = numpy.pow(a,b)

        ctx,queue = node.env.opencl_feature.context_queue("CPU")

        prg = cl.Program(ctx, """
            __kernel void sum(const int N, __global const float4 *a,
            __global const float4 *b, __global float4 *c)
            {
              int gid = get_global_id(0);
              for (int i = N*gid; i < N*gid+N; ++i) c[i] = pow(a[i] , b[i]);
            }
            """).build()
        mf = cl.mem_flags
        def rval():
            #print 'running OpenCL version'
            a = input_registers[0]
            b = input_registers[1]
            #output_registers[0] = a+b
            #return
            output_registers[0] = z = numpy.zeros_like(a)
            a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=a)
            b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=b)
            z_buf = cl.Buffer(ctx, mf.WRITE_ONLY | mf.USE_HOST_PTR, hostbuf=z)
            rval = prg.sum(queue, (2,), None, numpy.int64(len(a)/2/4), 
                    a_buf, b_buf, z_buf)
            #print rval
            rval.wait()
            #cl.enqueue_read_buffer(queue, z_buf, z).wait()

        rval.lazy=False
        return rval
pow_cl = PowCL()

swap_impls = True

@theano.tensor.opt.register_specialize
@theano.gof.local_optimizer([])
def add_to_addcl(node):
    if swap_impls:
        if node.op == T.pow:
            return [pow_cl(*node.inputs)]

