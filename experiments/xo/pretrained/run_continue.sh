export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/var/nix/profiles/gcc-5.4.0/lib64:$LD_LIBRARY_PATH
# python continue.py long --decoder-kind=mlp
python simple_continue.py long --decoder-kind=attn
python simple_continue.py long --decoder-kind=seq
python simple_continue.py long --decoder-kind=mlp