"""Evaluate sit: held-out quality, conditions, coverage and actual sampling cost.

This entry only loads frozen checkpoints; it never runs optimizer updates.
Use --split validation for tuning and --split test for the final report.
"""

from dl_utils.modern.evaluation import main

if __name__ == "__main__":
    main("sit", ["sit"])
