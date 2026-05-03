"""Per-node FedAvg entrypoint (Mode B - data-parallel + adapter averaging).

  $env:FUG_RANK="0"; python scripts/node_fedavg.py --rounds 5
"""
from fasturduguard.train.fedavg import main

if __name__ == "__main__":
    main()
