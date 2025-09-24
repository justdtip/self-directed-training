import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OPT_ROOT = pathlib.Path("/opt/azr")
if OPT_ROOT.exists() and str(OPT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPT_ROOT))
