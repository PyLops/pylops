{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
        ~{{ objname }}.{{ item }}
    {%- endfor %}

{% set allowed_classes = [
    "pylops.LinearOperator",
    "pylops.optimization.basesolver.Solver",
    "pylops.optimization.callback.Callbacks"
] %}

{% if fullname in allowed_classes %}
    {% for method in methods | reject("equalto", "__init__") %}
.. automethod:: {{ objname }}.{{ method }}
    {% endfor %}
{% endif %}


.. raw:: html

    <div style='clear:both'></div>


.. include:: backreferences/{{ fullname }}.examples

.. raw:: html

     <div style='clear:both'></div>

