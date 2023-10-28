# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:21:20 2023

@author: leoda
"""

import numpy as np
import matplotlib.pyplot as plt


def discrete_logistic_growth():
    r=2.9
    x=0.51
    iterations=100
    r=float(input('Growth rate here'))
    #x=float(input('Start value here'))
    #iterations=int(input('Number of iterations'))
    iterands=[x]
    for i in range(iterations):
        x=r*x*(1-x)
        iterands.append(x)
        print(x)
    return iterands

def plot_function(y):
    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True
    
    x = [i for i in range(len(y))]

    plt.title("Line graph")
    plt.plot(x, y, color="red")

    plt.show()

repetitions=int(input('How often repeat:'))
for i in range(repetitions):
    y=discrete_logistic_growth()
    plot_function(y)
