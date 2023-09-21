# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 18:19:42 2023

@author: leoda
"""
    

def triangular():
    start_triangle=285
    start_pentagon=165
    start_hexagon=143
    triangle=40755
    pentagon=40755
    hexagon=40755
    stop=False
    while not stop:
        start_triangle+=1
        triangle+=start_triangle
        print(triangle)
        if triangle>pentagon:
            start_pentagon+=1
            pentagon=start_pentagon*(3*start_pentagon-1)/2
        if triangle>hexagon:
            start_hexagon+=1
            hexagon=start_hexagon*(2*start_hexagon-1)
        if triangle==pentagon and triangle==hexagon:
            stop=True
    print(triangle)
        
    
triangular()