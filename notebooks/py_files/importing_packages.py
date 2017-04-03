import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
import re
import time
import gensim
import json
import random
import webbrowser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize                  
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
from gensim import corpora, models
