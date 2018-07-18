export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/var/nix/profiles/gcc-5.4.0/lib64:$LD_LIBRARY_PATH
python dair_search.py long --n-digits=1
python dair_search.py long --n-digits=3
python dair_search.py long --n-digits=5
python dair_search.py long --n-digits=7
python dair_search.py long --n-digits=9