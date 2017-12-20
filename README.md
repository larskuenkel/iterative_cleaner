# iterative_cleaner
RFI removal tool for pulsar archives.

Based on the surgical cleaner included in the coast_guard pipeline (https://github.com/plazar/coast_guard) by Patrick Lazarus. This pipeline is described in http://adsabs.harvard.edu/abs/2016MNRAS.458..868L.

The surgical cleaner was altered to in order to be more useful for LOFAR scintillation studies.

Two major changes were made:
  1. The cleaner uses an iterative approach for the template profile. This helps when the pulsar is masked by RFI in the original template profile.
  2. The detrending algorithm was removed. This feature may be reintroduced with different default parameters.
  
Usage:
  python iterative_cleaner.py -h
  
  python iterative_cleaner.py archive
