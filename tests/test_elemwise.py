import numpy, theano
from theano import tensor as T
import timeit

import theano_opencl

def test_0():

    N = 16*1000*10*1

    if 1:
        aval = abs(numpy.random.randn(N).astype('float32'))+.1
        bval = numpy.random.randn(N).astype('float32')
        a = T.fvector()
        b = T.fvector()
    else:
        aval = abs(numpy.random.randn(N))+.1
        bval = numpy.random.randn(N)
        a = T.dvector()
        b = T.dvector()

    f = theano.function([a,b], T.pow(a,b), mode='LAZY')
    theano_opencl.elemwise.swap_impls=False
    g = theano.function([a,b], T.pow(a,b), mode='LAZY')

    print 'ocl   time', timeit.Timer(lambda: f(aval, bval)).repeat(3,3)

    print 'gcc   time', timeit.Timer(lambda: g(aval, bval)).repeat(3,3)

    print 'numpy time', timeit.Timer(lambda: aval**bval).repeat(3,3)

    assert ((f(aval, bval) - aval**bval)**2).sum() < 1.1
    assert ((g(aval, bval) - aval**bval)**2).sum() < 1.1


if __name__ == '__main__':
    test_0()
