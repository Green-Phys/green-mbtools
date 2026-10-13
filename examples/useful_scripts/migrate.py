"""Reference wrapper: migrate a green-mbtools input.h5 to the current format.

This simply calls green_mbtools.mint.migrate. See that module for details:

    python -m green_mbtools.mint.migrate --input in.h5 --output out.h5 \
        [--int-path DIR ...] [--dm dm.h5] [--target 1.1.0] [--force]
"""
from green_mbtools.mint.migrate import _main

if __name__ == "__main__":
    _main()
