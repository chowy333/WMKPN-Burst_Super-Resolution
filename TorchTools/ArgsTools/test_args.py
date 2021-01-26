import os
import numpy as np
import shutil

class TestArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # datasets args
        parser.add_argument('--test_path', type=str,default='./data')

        self.args = parser.parse_args()

        return self.args

    def print_args(self):
        # print args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")
        print("==========     CONFIG END    =============")
    
