#!/bin/python3
#creating model

#preconfigured library
import imp
import logging, os, sys, time, threading
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

#customLibrary
from toolsetLib.configUtils import conf_load
from toolsetLib.datasetTool_factory import createDataset as create_dataset
from toolsetLib.loggingUtils import init_logger

