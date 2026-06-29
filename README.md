# FreeForm-modelling-LensFactory
Some initial scripts written during testing of free form lens modelling using LensFactory.

- FreeFormLens.jl is the file defining various helper functions for the lens object. To initialize a free lens, import this file and use the usual syntax to create a lens object in LensFactory.
- free_formlens_tests.ipynb : notebook for testing the various helper functions
- ModelFitter_tests.ipynb : Initial tests for modelling lenses.

There is also a modified input file for the Ares lensing data. "multimages.txt" houses the astrometric data for Ares cluster lens. It has been modified a bit for compatibility with the modelling pipeline of LensFactory. For example, I have added uncertainty columns (1 arcsec \sigma_x and \sigma_y for all images and error ellipse angle 0).
