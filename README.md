# EEG Complexity GUI

A repository containing pipelines for calculating some complexity measures from EEG data and a GUI for easy use. The measures included are Lempel-Ziv Complexity, Permutation Entropy, weighted Symbolic Mutual Information, and aspects of the power spectrum including FOOOF.

The easiest way to use the package is to go to "Releases" and download the latest release (Windows/Mac depending on your system) as a zip.
After extracting this, click the file called "Install", which will set up a localised virtual environment in this folder, so you have all the packages you need to run the programme.
This won't affect anything anywhere else on your computer as it is localised to this folder.
Then once this installation is complete, just click the "Start App" file and this will open the app.
Although the app opens in your browser, it all runs locally on your computer, i.e. without sharing the data online.

For those who want to look under the hood or use this code more flexibly, the python scripts are shared here.

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
