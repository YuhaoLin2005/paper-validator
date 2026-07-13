"""paper-validator — 5-layer governance audit sub-agent.

Importers: any Python code via `from paper_validator.layers import ...`
Callers: Path 3 native import users; any agent that embeds governance audit
Schema: no new data — sys.path fix for sibling imports
User verbatim: "三种集成路径都要自己先验证跑通"
"""

import sys, os
_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

__version__ = "1.0.0"
