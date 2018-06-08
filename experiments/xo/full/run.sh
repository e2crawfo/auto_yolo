export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/nix/var/nix/profiles/gcc-5.4.0/lib64:$LD_LIBRARY_PATH
# python yolo.py long --decoder-kind=attn
# python yolo.py long --decoder-kind=seq
# python yolo.py long --decoder-kind=mlp
python simple.py long --decoder-kind=attn
python simple.py long --decoder-kind=seq
python simple.py long --decoder-kind=mlp