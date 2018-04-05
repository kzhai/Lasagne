from .base import *
#
#
#
try:
	from .Xbase import *
	#from .Xlenet import *
	#from .Xmlp import *
	#from .XmlpDebug import *
except ImportError:
	raise ImportError("Could not load some of the XModules...")
	pass
#
#
#
#from .ctc import *
#from .dae import *
from .alexnet import *
from .elman import *
#from .fdn import *
from .lenet import *
from .mlp import *
#from .rbm import *
#from .rnn import *
from .snn import *
#from .vdn import *
