# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:15:50 2017

@author: Nat
"""
'''
go to the mesa project directory in anaconda prompt
run 'python supplyChain_viz.py'
'''
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule

import sys
sys.path.append('D:\\projects\\mesa\\')
from supplyChain import SupplyChainModel, Plant, Warehouse, Supplier, Transportation, Product, Order

def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 0}
    
    if isinstance(agent, Plant) or isinstance(agent, Warehouse) or \
        isinstance(agent, Supplier):
        portrayal['Color'] = 'blue'
        portrayal['r'] = 2
    
    if isinstance(agent, Transportation):
        portrayal['Color'] = 'green'
        portrayal['r'] = 2
    
    if isinstance(agent, Product):
        portrayal['Color'] = 'yellow'
        portrayal['Layer'] = 2
        portrayal['r'] = 0.2

    if isinstance(agent, Order):
        portrayal['Color'] = 'grey'
        portrayal['Layer'] = 1
        portrayal['r'] = 0.2
    
    return portrayal

grid = CanvasGrid(agent_portrayal, 100, 100, 800, 800)
chart1 = ChartModule([{'Label': 'average cost',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector1',
                    canvas_height = 400,
                    canvas_width = 800)
chart2 = ChartModule([{'Label': 'plant util',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector2',
                    canvas_height = 400,
                    canvas_width = 800)
        
server = ModularServer(SupplyChainModel,
                       [grid, chart1, chart2],
                       "Supply Chain Model",
                       {'width':100, 'height':100})
server.port = 8538
server.launch()

