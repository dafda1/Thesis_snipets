# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:58:21 2025

@author: dafda1
"""

# import pandas as pd

#%%

def write_dataframe_in_latex (dataframe, output_dir):    
    with open(output_dir, "w") as fobj:
        fobj.write(r"\hline" + "\n")
        
        fobj.write(dataframe.index.name)
        for col in dataframe.columns:
            fobj.write(" & " + col)
        fobj.write(r"\\" + "\n" + r"\hline" + "\n" + r"\hline" + "\n")
        
        for index in dataframe.index:
            fobj.write(index)
            for col in dataframe.columns:
                fobj.write(" & " + dataframe.at[index, col])
            fobj.write(r"\\" + "\n" + r"\hline" + "\n")
    
    return None