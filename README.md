# SuperWASP Tools

This repo contains some convenience classes and notebooks for working with the [SuperWASP Variable Stars](https://www.zooniverse.org/projects/ajnorton/superwasp-variable-stars) project data. This is mainly intended to be a starting point for ad-hoc analysis.

* __swasputils.py__ contains classes for loading various data products into Pandas data frames with methods for carrying out common tasks.
* __aggregated_classifications.ipynb__ loads the aggregated classification results and displays a bar chart of the number of subjects in each class.
* __display_siblings.ipynb__ displays all the lightcurves for a given SWASP ID. You can also give it a Zooniverse ID and it will display the lightcurves for subjects with the same SWASP ID.
* __zooniverse_sets.ipynb__ plots a bar chart of the number of active subjects per subject set in each workflow.

These scripts assume you have the various .csv and .dat files saved in a folder, given in `swasputils.DATA_LOCATION`.