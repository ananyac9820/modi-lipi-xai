"""REFERENCE BASELINES ONLY.

Nothing in this package is part of the Modi Lipi XAI pipeline or the web app.
It exists solely to answer the research question - at what training-set size
does handcrafted feature engineering stop being competitive with deep
learning? - by training a deliberately plain, untuned CNN on exactly the same
splits as the classical models. ``src/models.py``, ``src/predict.py`` and
``app/`` must never import from here.
"""
