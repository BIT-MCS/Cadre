import sys
# sys.path.append('/home/cst/wk/carla/AutoDriving_Carla')
# sys.path.append('/data2/wk/carla/AutoDriving_Carla')
# sys.path.append('/data2/wk/carla/VAE-pytorch/carla_perception')

import os
path = os.path.join(os.environ['CHALLENGE_DIR'], 'carla_perception')
sys.path.append(path)

# from demo import liteseg_mobilenet, liteseg_shufflenet
# from models import liteseg_mobilenet, liteseg_shufflenet
from Models import auto_trainer