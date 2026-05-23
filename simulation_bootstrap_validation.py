"""
Simulation-only validation entry point.

This file intentionally contains the synthetic-data path: it reads the bundled
synthetic workbook, simulates two annotation lists, trains q from reconstructed
states, runs bootstrap evaluation, and redraws the validation plots.

For real user-supplied lists and model predictions, use evaluate_two_lists_model.py.
"""

from run_list_state_simulation import run


if __name__ == "__main__":
    run()

