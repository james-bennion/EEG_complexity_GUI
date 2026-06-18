# complexity_shiny
A repository containing pipelines for calculating some complexity measures from EEG data and a GUI for easy use. The measures included are Lempel-Ziv Complexity, Permutation Entropy, weighted Symbolic Mutual Information, and aspects of the power spectrum including FOOOF.

This adapts functions from the concog-dreem-lib package, created by Max Hughes (https://pypi.org/project/concog-dreem-lib/).
The main changes include:
- Doesn't use MATLAB Engine - all in Python.
- Doesn't use custom channel groups - mainly calculates per-channel values, with some minor optional exceptions.
- Doesn't compare to original epochs and calculate gaps.
- Vectorised Permutation Entropy and wSMI for speedier calculation, especially for large number of channels.
- Added the new metric Aperiodic-Adjusted Power.
- Added centralised processing tracker
- Added shiny-based GUI for easy use, particularly for users less familiar with Python

Thanks to Max Hughes for the original package and Tristan Bekinschtein, Rianne Haartsen, and Emily Jones for their supervision.
