import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches




physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", physical_devices[0].name)