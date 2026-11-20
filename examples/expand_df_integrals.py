"""Expand SG factors into a fresh legacy directory in the original input pair order."""

import argparse
from green_mbtools.mint.integral_symmetry_io import expand_archive

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", required=True)
parser.add_argument(
    "--input", required=True, help="Identified copy of the original input.h5"
)
parser.add_argument("--out", required=True)
parser.add_argument("--chunk-size", type=int, default=32)
args = parser.parse_args()
expand_archive(args.archive, args.input, args.out, chunk_size=args.chunk_size)
