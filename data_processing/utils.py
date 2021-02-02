import pandas as pd
def print_full(x):
    """Print complete df"""
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def display_full(x):
    """Display complete df"""
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', 100)
    display(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
        
def joinnames(*names): 
    return '_'.join([name for name in names if name])
