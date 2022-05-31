#!/bin/python3
import os, sys, time, threading, pprint, logging, yaml
from easydict import EasyDict as ED
import toolsetLib.defaultConfUtils
#load default configuration
conf = toolsetLib.defaultConfUtils.conf
# merge dict b into dict a with source 
def _mergedict_a_into_b(a, b):
    if type(a) is not ED:
        return
    for k, v in a.items():
        # a must specify keys that are in b
        if type(v) is ED:
            try:
                _mergedict_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
# load configuration from file
def conf_from_file(filename):
    with open(filename, 'r') as f:
        #yaml_cfg = ED(yaml.load(f)) -> this wont work load() missing 1 required positional argument: 'Loader'
        yaml_cfg = ED(yaml.load(f, Loader=yaml.FullLoader))
    _mergedict_a_into_b(yaml_cfg, conf)
    logging.info("Config:\n"+pprint.pformat(conf))
    return conf

# load configuration from command line environment
def conf_load():
    filename = "./train/pose_cfg.yaml" #change this if you change the config file name
    if 'POSE_PARAM_PATH' in os.environ:
        filename = os.environ['POSE_PARAM_PATH'] + '/' + filename
    return conf_from_file(filename)

#main thread for this module
if __name__ == "__main__":
    print(conf_load())

