import glob
import pickle
from AREA import *

classifiers = pickle.load(open('basic_classifiers.pkl','rb'))
filepaths = glob.glob('../TUT-acoustic-scenes-2016-development/evaluation_setup/fold*_evaluate.txt')

test_info = {}
for path in filepaths:
   test_info[path[57:62]] = extract_info(path)

test_results = {}
for info, data in test_info.items():
    test_results[info] = classifiers[info].classify(data)
