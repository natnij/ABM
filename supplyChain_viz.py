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

import os
os.chdir('D:\\projects\\mesa')
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
chart1 = ChartModule([{'Label': 'cost per product',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)
chart2 = ChartModule([{'Label': 'plant utilization',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)
chart3 = ChartModule([{'Label': 'CN warehouse',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)
chart4 = ChartModule([{'Label': 'EU order fulfillment',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)        
chart5 = ChartModule([{'Label': 'LA warehouse',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)
chart6 = ChartModule([{'Label': 'EU warehouse',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)
chart7 = ChartModule([{'Label': 'ME warehouse',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800)
chart8 = ChartModule([{'Label': 'LA order fulfillment',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800) 
chart9 = ChartModule([{'Label': 'ME order fulfillment',
                      'Color': 'Black'}],
                    data_collector_name = 'datacollector',
                    canvas_height = 400,
                    canvas_width = 800) 
        
server = ModularServer(SupplyChainModel,
                       [grid, chart1, chart2, chart3,
                        chart5, chart6, chart7,
                        chart4, chart8, chart9],
                       "Supply Chain Model",
                       {'width':100, 'height':100})
server.port = 8521
server.launch()

