"""Per-node training entrypoint (Mode A).

  $env:FUG_RANK="0"; python scripts/node_train.py
  $env:FUG_RANK="1"; python scripts/node_train.py
"""
from fasturduguard.train.run import main


if __name__ == "__main__":
    main()
