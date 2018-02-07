Links and Hints
===============

.. todolist::

Sphinx
------
* http://www.sphinx-doc.org/en/stable/domains.html#cross-referencing-python-objects
* http://www.sphinx-doc.org/en/stable/markup/inline.html

MathJax:
    * http://www.sphinx-doc.org/en/stable/ext/math.html
    * http://docs.mathjax.org/en/latest/tex.html#supported-latex-commands

Git
---
To update the pca repo, execute::

  git subtree push --prefix python/pca pca master

or for Antarctica::

  git subtree push --prefix notebooks/Antarctica ant master

(from top level directory, i.e. ``code``).

* https://lostechies.com/johnteague/2014/04/04/using-git-subtrees-to-split-a-repository/
* http://www.paulwhippconsulting.com/blog/splitting-a-git-repository-into-two-projects-and-reintegrating-them/
* https://www.atlassian.com/blog/git/alternatives-to-git-submodule-git-subtree

.. include:: ../python/pca/docs/index.rst

Data
====
.. automodule:: python.data
   :members:

Various linear stats
====================
.. automodule:: python.linear

Helpers
=======
             
.. automodule:: python.helpers
   :members:

Geo
===

.. automodule:: python.geo
   :members:

Plotting
========

.. automodule:: python.plots
   :members:


