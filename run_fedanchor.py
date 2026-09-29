"""
Press the Run button on this file in VS Code to train and evaluate FedAnchor v2 on MNIST.

Prints one line per round with:
  kNN = kNN classification accuracy of the 2D embedding (0.91 = 91%)
  T   = trustworthiness,  C = continuity  (how well the 2D map keeps the original neighbourhoods)
"""
import os
import sys
import runpy

# ---- settings you can change ----
ROUNDS = 20          # 20 = full benchmark; use 5 for a quick demo
PARTITION = "iid"    # "iid" or "dirichlet0.1" (non-IID: each client gets mostly different digits)
# ---------------------------------

ROUNDS = int(os.environ.get("ROUNDS", ROUNDS))
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)                      # works no matter which folder VS Code runs from
sys.path.insert(0, ROOT)

print(f"Running FedAnchor v2: {ROUNDS} rounds, partition={PARTITION} (MNIST downloads automatically the first time)\n", flush=True)
sys.argv = ["run_v2_benchmark.py", "anchor", str(ROUNDS), PARTITION]
runpy.run_path(os.path.join(ROOT, "fedanchor", "run_v2_benchmark.py"), run_name="__main__")
print("\nDone. The last line above is the final result.")
