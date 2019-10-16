from fastai.vision import *
from fastai.basic_train import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'