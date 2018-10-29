#! /usr/bin/env python
# -*- coding:utf-8 -*-

import codecs
import sys
import os
import json
import random
import numpy as np
from functools import reduce
import jieba

data_dir = 'data'
save_dir = 'save'

if not os.path.exists(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
