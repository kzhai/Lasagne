from .alexnet import *
from .alexnetd import *
from .base import *
from .debugger import *
# from .fdn import *
from .lenet import *
from .mlp import *
from .snn import *

# from .vdn import *

try:
	from .Xdlenet import *
	from .Xdmlp import *
except ImportError:
	print "Some of the XModules are not found..."
	pass
