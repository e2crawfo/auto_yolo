export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/var/nix/profiles/gcc-5.4.0/lib64:$LD_LIBRARY_PATH
python air_search.py build --n-digits=1
# python air_search.py build --n-digits=2
# python air_search.py build --n-digits=3
# python air_search.py build --n-digits=4
# python air_search.py build --n-digits=5