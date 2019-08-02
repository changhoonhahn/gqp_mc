'''

some utility functions 

'''
import os


def check_env(): 
    if os.environ.get('GQPMC_DIR') is None: 
        raise ValueError("set $GQPMC_DIR in bashrc file!") 
    return None


def dat_dir(): 
    return os.environ.get('GQPMC_DIR') 

