Google Earth Engine
===================

.. js:autofunction:: format_table

Python API
----------

First, initialize earth engine::

  import ee
  ee.Initialize()

Get some metadata on an ImageCollection::

  s1 = ee.ImageCollection('COPERNICUS/S1_GRD')

  # total size
  s1.size().getInfo()

  # properties ('columns'?)
  s1.propertyNames().getInfo()

The ``getInfo()`` function seems to be what submits the query to the actual earth engine server to obtain a result.

Get info on one image from collection::

  s1.toList(1).get(0).getInfo()
