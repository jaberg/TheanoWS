from theano.gof.opt import Optimizer
import theano
from theano import tensor as T

import pyopencl as cl

class OpenClFeature(object):

    def __init__(self):
        self._cpu_context = None
        self._cpu_queue = None

    def on_attach(self, env):
        assert not hasattr(env, 'opencl_feature')
        env.opencl_feature = self

        for node in env.toposort():
            self.on_import(env, node)

    def get_cpu_context(self):
        if self._cpu_context is None:
            self._cpu_context = cl.Context(dev_type=cl.device_type.CPU)
        return self._cpu_context

    def get_cpu_queue(self):
        if self._cpu_queue is None:
            self._cpu_queue = cl.CommandQueue(self.cpu_context)
        return self._cpu_queue

    cpu_context = property(get_cpu_context)
    #gpu_context = property(get_gpu_context)
    cpu_queue = property(get_cpu_queue)
    #gpu_queue = property(get_gpu_queue)


    def context_queue(self, devtype):
        if devtype=='CPU':
            return self.cpu_context, self.cpu_queue
        raise NotImplementedError(devtype)

    def on_import(self, env, node):
        pass

    def on_change_input(self, env, node, i, r, new_r):
        pass

class OpenCLOptimizer(Optimizer):
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(OpenClFeature())

    def apply(self, env):
        pass

# -1 should make it run right before the first merge
theano.compile.mode.optdb.register('OpenCL_Opt',
        OpenCLOptimizer(), -1, 
        'fast_run', 'fast_compile')


from . import elemwise
