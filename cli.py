#!/usr/bin/env python

import argparse

import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video.")
    parser.add_argument("file")
    parser.add_argument("--filters")
    args = parser.parse_args()
    analyze.analyze(args.file, args.filters)
