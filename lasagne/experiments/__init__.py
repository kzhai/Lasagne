from .alexnet import *
#from .alexnetd import *
from .argument import *
from .base import *
from .debugger import *
# from .fdn import *
from .lenet import *
from .mlp import *
from .snn import *

# from .vdn import *

try:
	from .lenetA import *
	from .mlpA import *
except ImportError:
	raise ImportError("Could not load some of the XModules...")
	pass
