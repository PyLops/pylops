.. _faq:

Frequenty Asked Questions
=========================

**1. Can I visualize my operator?**

Yes, you can. Every operator has a method called ``todense`` that will return the dense matrix equivalent of
the operator. Note, however, that in order to do so we need to allocate a ``numpy`` array of the size of your
operator and apply the operator ``N`` times, where ``N`` is the number of columns of the operator. The allocation can
be very heavy on your memory and the computation may take long time, so use it with care only for small toy
examples to understand what your operator looks like. This method should however not be abused, as the reason of
working with linear operators is indeed that you don't really need to access the explicit matrix representation
of an operator.


**2. Can I have an older version of** ``cupy`` **installed in my system (** ``cupy-cudaXX<10.6.0``)?**

Yes. Nevertheless you need to tell PyLops that you don't want to use its ``cupy``
backend by setting the environment variable ``CUPY_PYLOPS=0``.
Failing to do so will lead to an error when you import ``pylops`` because some of the ``cupyx``
routines that we use are not available in earlier version of ``cupy``.