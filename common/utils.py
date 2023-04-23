import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.requset
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from iopath.common.download import download
from iopath.common.file_io import file_lock, g_pathmgr