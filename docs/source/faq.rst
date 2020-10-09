.. _faq:

Frequenty Asked Questions
=========================

**1. Can I visualize my operator?**

Yes, you can. Every operator has a method called ``todense`` that will return the dense matrix equivalent of
the operaotor. Note, however, that in order to do so we need to allocate a numpy array of the size of your
operator and apply the operator N times, where N is the number of columns of the operator. The allocation can
be very heavy on your memory and the computation may take long time, so use it with care only for small toy
examples to understand what your operator looks like. This method should however not be abused, as the reason of
working with linear operators is indeed that you don't really need to access the explicit matrix representation
of an operator.
