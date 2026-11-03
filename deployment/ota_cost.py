#!/usr/bin/env python3
"""Calculate nominal OTA transfer and SIM cost for the specified strategies."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-kb", type=float, default=280)
    parser.add_argument("--trucks", type=int, default=85)
    parser.add_argument("--canary", type=int, default=10)
    parser.add_argument("--rate-per-mb", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 < args.canary < args.trucks:
        raise ValueError("Canary count must be between zero and total fleet.")
    mb = args.model_kb / 1000
    print(f"Assumption: decimal MB; {args.model_kb:g} KB/model; ₹{args.rate_per_mb:g}/MB")
    for name, count in (
        ("Full replacement", args.trucks),
        ("Canary initial phase", args.canary),
        ("Canary remaining phase", args.trucks - args.canary),
        ("Shadow candidate to all trucks", args.trucks),
    ):
        transfer = count * mb
        print(f"{name}: {count} trucks, {transfer:.3f} MB, ₹{transfer*args.rate_per_mb:.3f}")


if __name__ == "__main__":
    main()
