{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    .. rubric:: Methods

    .. autosummary::
    {% for item in methods %}
        ~{{ objname }}.{{ item }}
    {%- endfor %}

{#    .. rubric:: Attributes#}
{##}
{#    .. autosummary::#}
{#    {% for item in attributes %}#}
{#        ~{{ objname }}.{{ item }}#}
{#    {%- endfor %}#}

.. raw:: html

     <div style='clear:both'></div>


.. include:: backreferences/{{ fullname }}.examples

.. raw:: html

     <div style='clear:both'></div>


{#{% for item in methods %} #}
{#{% if item != '__init__' %} #}
{#.. automethod:: {{ objname }}.{{ item }} #}
{#{% endif %} #}
{#{% endfor %} #}

{#.. raw:: html #}
{# #}
{#     <div style='clear:both'></div> #}

