from src.self_awareness.networks.squeezenet import Squeezenet
from src.self_awareness.networks.squeezenet import Squeezenet_Localization
from src.self_awareness.networks.simplenet import Simplenet_Localization
from src.self_awareness.networks.vggnet import Vggnet_Localization
import src.self_awareness.networks.resnet
import src.self_awareness.networks.densenet

catalogue = dict()


def register(cls):
    catalogue.update({cls.name: cls})


register(Squeezenet)
register(Squeezenet_Localization)
register(Simplenet_Localization)
register(Vggnet_Localization)