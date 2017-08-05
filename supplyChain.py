# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:38:59 2017

@author: Nat
"""
from mesa import Agent, Model
from mesa.time import RandomActivation
import random
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
import pandas as pd
import os
os.chdir('D:\\projects\\mesa\\output')
from matplotlib import pyplot as plt

class Transportation(Agent):
    def __init__(self, unique_id, model, specialFreight = False, speed = 1, 
                 priceIncreaseFactor = 10, reliability = 0.9, 
                 weightCapacity = 1000, volumeCapacity = 1000, singleTripMode = True):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.weightCapacity = weightCapacity # as absolute units
        self.volumeCapacity = volumeCapacity # as absolute units
        self.utilizedWeight = 0  # as absolute units
        self.utilizedVolume = 0  # as absolute units
        self.reliability = reliability # e.g. 0.9 is 90%
        self.unitWeightPrice = 1 
        self.unitVolumePrice = 1
        self.priceIncreaseFactor = priceIncreaseFactor
        self.volumeDiscount = {0:0, 50:0.1, 80:0.15, 95:0.2} # if utilized capacity > percentage, then price is discounted by decimal
        self.minUtilizationToSail = 50 # minimum utiliation percentage to sail, e.g. 50 is 50%
        self.fixedIntervalToSail = 7 # only sail every fixed time units
        self.speed = speed # grid to travel per time unit
        self.origin = (0, 0) # to be defined when created according to plant location
        self.destination = (1, 1) # to be defined when created according to customer location
        self.intransit = False
        self.specialFreight = specialFreight
        self.listOfProducts = []
        self.totalTransportCost = 0
        self.nrOfTrips = 0
        self.originDelay = 0 # of delays in origin
        self.transitDelay = 0
        self.singleTripMode = singleTripMode # if no return trip designed in the system, then True
    
    def step(self):
        if self.pos == self.origin: # load if transport is not moving in origin
            self.load()
        if self.pos == self.destination: # unload if transport is not moving in destination
            self.unload()
        if self.intransit is False:
            self.setsail()
        self.move()
    
    def load(self):     
        tmp = list(self.model.grid[self.origin[0]][self.origin[1]])
        listOfProducts = [product for product in tmp if isinstance(product, Product) and product.intransit is False]
        if len(tmp) == 0:
            return
        
        loadingSequence = sortAgent(listOfProducts, 'priority', reverse = False)
      
        for i in loadingSequence:
            product = listOfProducts[i]
            if product.weight * product.quantity > self.weightCapacity - self.utilizedWeight:
                return
            self.listOfProducts.append(product)
            self.utilizedWeight = self.utilizedWeight + product.weight * product.quantity
            self.utilizedVolume = self.utilizedVolume + product.volume * product.quantity
            product.intransit = True
        
    def unload(self):
        # find warehouse in neighborhood
        content = self.model.grid.get_cell_list_contents(self.get_neighborhood())
        whs = [agent for agent in content if isinstance(agent, Warehouse)][0]
        for product in self.listOfProducts:
            product.warehouse = whs
            self.model.grid.move_agent(product, whs.pos)
            product.availableInStorage = True 
            whs.utilization = whs.utilization + \
                product.volume / whs.stackLayer * product.quantity
            whs.listOfProducts.append(product)
            self.calTransportCost() # calculate cost upon arrival
            self.nrOfTrips = self.nrOfTrips + 1
            self.intransit = False
        
        if self.singleTripMode:
            self.resetPosition(self.origin, self.destination)
        else:
            # reverse privious origin and destination pair for the new trip
            self.resetPosition(self.destination, self.origin)

    def setsail(self):
        reachedWeightLimit = self.utilizedWeight / self.weightCapacity * 100 > self.minUtilizationToSail
        reachedVolumeLimit = self.utilizedVolume / self.weightCapacity * 100 > self.minUtilizationToSail
        isScheduledDate = np.mod(self.model.schedule.time, self.fixedIntervalToSail) == 0
        
        if  (self.specialFreight) or (isScheduledDate and (reachedWeightLimit or reachedVolumeLimit)):        
            self.intransit = True
           
    def move(self):
        if self.intransit is False:
            return
        possible_steps = self.get_neighborhood()
        shortestDistance = np.argmin(self.calDistance(possible_steps))
        new_position = possible_steps[shortestDistance]
        if random.randrange(0, 100) <= self.reliability * 100:
            self.model.grid.move_agent(self, new_position)
            self.pos = new_position
            for product in self.listOfProducts:
                product.move(new_position)
        else:
            if self.pos == self.origin:
                self.originDelay = self.originDelay + 1
                self.intransit = False # if not leaving origin: reset intransit to False
            else:
                self.transitDelay = self.transitDelay + 1

    def calDistance(self, listOfCells):
        distance = []
        for cell in listOfCells:
            distance.append(np.sqrt(np.power(cell[0] - self.destination[0],2) + 
                            np.power(cell[1] - self.destination[1],2)))
        return distance

    def calTransportCost(self):
        if self.specialFreight:
            tmpUnitWeightPrice = self.unitWeightPrice * self.priceIncreaseFactor
            tmpUnitVolumePrice = self.unitVolumePrice * self.priceIncreaseFactor
        else:
            tmpUnitWeightPrice = self.unitWeightPrice
            tmpUnitVolumePrice = self.unitVolumePrice
        # calculate price after volume discount
        quantityStep = np.array(sorted(self.volumeDiscount.keys()))
        utilized = max(self.utilizedVolume / self.volumeCapacity * 100, self.utilizedWeight / self.weightCapacity * 100)
        discount = self.volumeDiscount.get(quantityStep[sum(utilized > quantityStep) - 1])
        tmpUnitWeightPrice = tmpUnitWeightPrice * (1 - discount)
        tmpUnitVolumePrice = tmpUnitVolumePrice * (1 - discount)        
        
        for product in self.listOfProducts:
            product.transportCost = max(product.weight * product.quantity * tmpUnitWeightPrice, 
                                        product.volume * product.quantity * tmpUnitVolumePrice)
            self.totalTransportCost = self.totalTransportCost + product.transportCost
            
    def get_neighborhood(self):
        return self.model.grid.get_neighborhood(
                self.pos, moore = True, include_center = False)
        
    def resetPosition(self, origin, destination):
        # to reuse the vessel for simulation if there is no return trip designed in the system
        self.origin = origin
        self.destination = destination
        self.intransit = False
        self.specialFreight = False
        self.listOfProducts = []
        self.utilizedWeight = 0  # as absolute units
        self.utilizedVolume = 0  # as absolute units
        self.model.grid.move_agent(self, self.origin)
        
#%%        
class Product(Agent):
    def __init__(self, unique_id, model, quantity = 1, supplier = None,
                 weight = 1, volume = 1):
        super().__init__(unique_id, model)
        # intrinsic product characteristics
        self.name = None # product code. the unique id is also with batch information.
        self.weight = weight # unit weight
        self.volume = volume # unit volume
        self.value = 100 # unit value
        self.supplier = supplier
        self.decayRate = 0.001 # absolute value every time unit to subtract from shelfLifeLeft. 
                              # when down to zero the product should be scrapped
        # supply chain characteristics
        self.order = None
        self.quantity = quantity
        self.priority = 999 # the smaller the number, the higher the loading priority
        self.shelfLifeLeft = 1
        self.pos = None
        self.intransit = False
        self.warehouse = None
        self.active = True # if scrapped or delivered, then active is False
        self.availableInStorage = False
        self.bookedForCustomer = False
        self.scrapped = False
        # financial characteristics
        # all are total costs of the batch
        self.purchaseCost = self.calPurchaseCost()
        self.transportCost = 0
        self.capitalCost = 0
        self.storageCost = 0
        self.scrapCost = 0

    def step(self):
        if not self.active:
            return
        self.shelfLifeLeft = self.shelfLifeLeft - self.decayRate
        if self.shelfLifeLeft == 0: 
            self.scrapped = True
            self.active = False
            self.availableInStorage = False
            self.calScrapCost()
        self.calCapitalCost()
        
    def move(self, new_position):
        self.model.grid.move_agent(self, new_position)
        self.pos = new_position

    def split(self, partialQuantity):
        if partialQuantity >= self.quantity:
            return
        product = Product(self.name + str(self.model.num_product), self.model, 
                          quantity = partialQuantity, supplier = self.supplier, 
                          weight = self.weight, volume = self.volume)
        product.name = self.name
        product.value = self.value
        product.decayRate = self.decayRate # absolute value every time unit to subtract from shelfLifeLeft. 
                              # when down to zero the product should be scrapped
        # supply chain characteristics
        product.order = self.order
        product.priority = self.priority# the smaller the number, the higher the loading priority
        product.shelfLifeLeft = self.shelfLifeLeft
        product.pos = self.pos
        product.intransit = self.intransit
        product.warehouse = self.warehouse
        product.active = self.active # if scrapped or delivered, then active is False
        product.availableInStorage = self.availableInStorage
        product.bookedForCustomer = self.bookedForCustomer
        product.scrapped = self.scrapped
        # financial characteristics
        # all are total costs of the batch
        product.purchaseCost = self.purchaseCost * partialQuantity / self.quantity
        product.transportCost = self.transportCost * partialQuantity / self.quantity
        product.capitalCost = self.capitalCost * partialQuantity / self.quantity
        product.storageCost = self.storageCost * partialQuantity / self.quantity
        product.scrapCost = self.scrapCost * partialQuantity / self.quantity
        
        self.quantity = self.quantity - partialQuantity
        self.model.schedule.add(product)
        self.model.num_product = self.model.num_product + 1
        self.model.grid.place_agent(product, self.pos)
        
        return [self, product]

    def calPurchaseCost(self):
        # if supplier is None, then no purchase cost. 
        if self.supplier is None:
            return 0
        else:
            maxDiscountLevel = np.max([x[1] for x in self.supplier.listOfProducts.keys() if x[0] == self.name])
            volumeDiscount = np.min(maxDiscountLevel, int(self.quantity / 100))  # every multiply of 100 units gets a discount
            tbl = self.supplier.listOfProducts
            unitPrice = float(tbl.loc[(tbl['MATERIAL_CODE'] == self.name) & 
                               (tbl['volumeDiscountCode'] == volumeDiscount), 'unitPrice'])
            return unitPrice * self.quantity
        
    def calCapitalCost(self):
        self.capitalCost = self.capitalCost + \
                    self.value * self.quantity * self.model.rules.companyInterestRate
        
    def calScrapCost(self):
        return self.value * self.quantity
#%%    
class Warehouse(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.defaultPrice = 1
        self.overCapacityPrice = 1.2
        self.unitPrice = self.defaultPrice
        self.capacity = 10000
        self.utilization = 0
        self.priceIncreaseFactor = 2
        self.stackLayer = 2
        self.listOfProducts = []
        self.totalStorageCost = 0
    
    def step(self):
        if self.utilization > self.capacity:
            self.unitPrice = self.defaultPrice
        else: 
            self.unitPrice = self.overCapacityPrice
        self.calStorageCost()
    
    def calStorageCost(self):
        for product in self.listOfProducts:
            if product.active:
                product.storageCost = self.unitPrice / self.stackLayer * \
                                      product.volume * product.quantity
                self.totalStorageCost = self.totalStorageCost + product.storageCost        
#%%    
class Supplier(Agent):
    def __init__(self, unique_id, model, listOfProducts = None):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.listOfProducts = listOfProducts.copy()
        # every multiply of 10 units gets a discount

class MarketPlanner(Agent):
    def __init__(self, unique_id, model, whs):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.responsibleProductGroup = []
        self.responsibleRegion = []
        self.reliability = 0.9
        self.marketplanDir = None # folder where all historical plans are
        self.currentMarketPlan = None
        self.whs = whs
    
    def step(self):
        self.updatePlan()
        self.passOrder()
    
    def updatePlan(self):
        '''
        update market plan.
        forecast algorithm can be plugged in here as one option.
        '''
        self.currentMarketPlan = pd.read_csv(self.marketplanDir + self.unique_id + '_' + 
                                             str(self.model.schedule.time - 1) + 'csv')
        # here run whatever algorithm
        
        self.currentMarketPlan.to_csv(self.marketplanDir + self.unique_id + '_' + 
                                             str(self.model.schedule.time) + 'csv')
        
    def kpi(self):
        '''
        get business KPI value 
        '''
        
    def passOrder(self):        
        '''
        split order and give order to corresponding production scheduler
        '''
        for region in self.responsibleRegion:
            eval(region) = []
        for row in range(self.currentMarketPlan.shape[0]):
            record = pd.DataFrame([self.currentMarketPlan.iloc[row,:]])
            if record['RPD'] < self.model.schedule.time:
                continue
            if np.array(record['originRegion'])[0] in self.responsibleRegion:
                if np.array(record['forecastProduct'])[0] in self.responsibleProductGroup:
                    eval(np.array(record['originRegion'])[0]) = eval(np.array(record['originRegion'])[0]).append([row])
        
        plants = self.model.setup.plant.copy()
        for region in self.responsibleRegion:
            x = int(plants.loc[(plants['plantRegion'] == region), 'x'])
            y = int(plants.loc[(plants['plantRegion'] == region), 'y'])
            plan = self.currentMarketPlan[self.currentMarketPlan.index.isin(eval(region))]
            content = self.model.grid[x][y]
            productionscheduler = [x for x in content if isinstance(x, ProductionScheduler)]
            productionscheduler = productionscheduler[0]
            productionscheduler.marketPlan = pd.concat([productionscheduler.marketPlan,plan], axis = 0)

class Order(Agent):
    def __init__(self, unique_id, model, orderLines):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.listOfProducts = []
        self.orderLines = orderLines.copy() # list of product names and quantities
        self.originRegion = np.array(self.getFirstLine(orderLines)['origin'])[0]
        self.destRegion = np.array(self.getFirstLine(orderLines)['dest'])[0]
        self.creationDate =  np.array(self.getFirstLine(orderLines)['orderCreateDate'])[0]
        self.CRD =  np.array(self.getFirstLine(orderLines)['CRD'])[0]
        self.RPD = None
        self.ASD = None
        self.ATA = None
        self.productAllocatedToOrder = False
        self.fulfilled = False
        self.onTimeInFull = False
        self.specialFreight = False
        
    def getFirstLine(self, orderLines):
        if orderLines.shape[0] == 1:
            return orderLines
        else:
            return pd.DataFrame([orderLines.iloc[0,:]])
    
    def fulfillOrder(self):
        self.fulfilled = True
        self.ATA = self.model.schedule.time
        if self.ATA <= self.CRD:
            self.onTimeInFull = True
        for product in self.listOfProducts:
            product.active = False            
            product.availableInStorage = False
        
    def step(self):
        if self.fulfilled:
            return
        self.productAllocatedToOrder = np.sum(self.orderLines['quantity']) == 0
        if (self.productAllocatedToOrder) and (self.ASD is None):
            # set actual shipping date to when product becomes available for the order
            self.ASD = self.model.schedule.time
        availability = [x.availableInStorage for x in self.listOfProducts]
        if (self.productAllocatedToOrder) and (np.all(availability)):
            self.fulfillOrder()
#%%
class ProductionScheduler(Agent):
    def __init__(self, unique_id, model, plantAgent):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.productionPlan = []
        self.plant = plantAgent
        self.marketPlan = []
        self.productionPlanDir = None
        self.reliability = 0.9
        self.customerOrders = []
    
    def readInOrders(self):
        
    
    def updateProductionPlan(self):
        self.productionPlan = self.model.setup.productionPlan.copy()
        self.productionPlan.to_csv(self.productionPlanDir + self.unique_id + '_' + 
                                             str(self.model.schedule.time) + 'csv')
        self.plant.productionPlan = self.productionPlan.copy()
    
    def step(self):
        self.readInOrders()
        self.updateProductionPlan()
        self.marketPlan = [] # reset marketPlan to prepare for receiving new ones in next time unit
        
#%%        
class Plant(Agent):
    def __init__(self, unique_id, model, whsAgent, dailyCapacity = 2000, reliability = 0.95):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.setup = self.model.setup
        self.whs = whsAgent
        self.dailyCapacity = dailyCapacity
        self.reliability = reliability
        self.averageUtilization = 0
        self.utilization = 0
        self.productionPlan = []
    
    def step(self):
        self.produce()
        self.calUtilization()
    
    def produce(self):
        if len(self.productionPlan) == 0:
            return
        productionPlan = self.productionPlan[(self.productionPlan['plannedProduction'] <= self.model.schedule.time) &
                                        (self.productionPlan['actualProduction'] == -1)]
        productionPlan.sort_values(by = 'priority', ascending = True, inplace = True)
        planIndex = productionPlan.index.tolist()
        for i in range(productionPlan.shape[0]):
            if random.randrange(0, 100) > self.reliability * 100:
                continue
            line = pd.DataFrame([productionPlan.iloc[i,:]])
            productName = np.array(line['MATERIAL_CODE'])[0]
            record = self.setup.product[self.setup.product['MATERIAL_CODE'] == productName]
            quantity = int(line.loc[:,'quantity'])
            if self.utilization + quantity >= self.dailyCapacity:
                return

            orderID = np.array(line['customerOrder'])[0]
            product = Product(productName + str(self.model.num_product), self.model, 
                              quantity = quantity,
                              supplier = None, weight = int(record.weight), volume = int(record.volume))
            product.name = productName
            if not orderID == 'NA':
                content = self.model.grid.get_cell_list_contents(self.pos)
                for order in content:
                    if isinstance(order, Order) and order.unique_id == orderID:
                        product.order = order
                        product.bookedForCustomer = True
                        product.priority = product.order.RPD
                        break

            self.model.schedule.add(product)
            self.model.grid.place_agent(product, self.pos)            
            self.whs.utilization = self.whs.utilization + product.volume / self.whs.stackLayer * product.quantity                      
            self.whs.listOfProducts.append(product)
            self.model.num_product = self.model.num_product + 1
            self.utilization = self.utilization + product.quantity
            self.productionPlan.loc[productionPlan.index == planIndex[i], 'actualProduction'] = self.model.schedule.time
    
    def calUtilization(self):
        if self.model.schedule.time > 0:
            self.averageUtilization = (self.averageUtilization * (self.model.schedule.time - 1) + self.utilization) / self.model.schedule.time
        # reset
        self.utilization = 0
        
    def buy(self, productName, productQuantity, supplierAgent):
        record = self.setup.product[self.setup.product['MATERIAL_CODE'] == productName]
        product = Product(productName + str(self.model.num_product), self, quantity = productQuantity, 
                        supplier = supplierAgent, weight = record.weight, volume = record.volume)
        product.name = productName
        self.model.schedule.add(product)
        self.model.grid.place_agent(product, supplierAgent.pos)
        self.model.num_product = self.model.num_product + 1    
#%%
class Rules(object):
    def __init__(self):
        self.companyInterestRate = 0.000246575342465753 #absolute value as in 9% / 365 days, assuming every time unit is one day.
        self.marketplanDir = 'D:\\projects\\mesa\\files\\'
        self.productionPlanDir = 'D:\\projects\\mesa\\files\\'

class Setup(object):
    def __init__(self):
        self.whs = pd.DataFrame([['CN',20,60],['EU',50,10],
                                 ['ME',70,35],['LA',95,95]], 
                                columns = ['REGION_EN_NAME', 'x', 'y'])
        self.plant = pd.DataFrame([['CN',20,60,'CN',2000]],
                        columns = ['plantRegion','x','y','whs_name','capacity'])
        self.originDestination = \
            pd.DataFrame([['CN','EU',19,59,49,11],
                          ['CN','ME',21,59,69,35],
                          ['CN','LA',21,61,94,94], 
                          ['supplier1','CN',29,70,21,60],
                          ['supplier2','CN',39,60,21,60]],
               columns = ['origin','dest','originX','originY','destX','destY'])
        self.product = pd.DataFrame([['A','supplier1',1,1],
                             ['B', None,1,1], ['C',None,1,1]], 
                             columns = ['MATERIAL_CODE','supplierName','weight','volume'])
        self.supplier = pd.DataFrame([['supplier1', 30,70],['supplier2',40,60]],
                                     columns = ['supplierName','x','y'])
        self.supplierCatalog = \
            pd.DataFrame([['supplier1','A',0,1],
                          ['supplier1','A',1,0.95],
                          ['supplier1','A',2,0.9],
                          ['supplier1','A',3,0.8],
                          ['supplier1','D',0,1],
                          ['supplier1','D',1,0.7]], 
            columns = ['supplierName', 'MATERIAL_CODE','volumeDiscountCode','unitPrice'])
        self.orderLines = pd.DataFrame([[1, 60,'CN','EU','B',200],
                                        [1, 50,'CN','ME','C',200],
                                        [1, 60,'CN','EU','A',300],
                                        [5, 65,'CN','EU','B',100],
                                        [10,70,'CN','EU','A',100],
                                        [15,80,'CN','EU','C',200],
                                        [16,45,'CN','ME','B',500],
                                        [20,70,'CN','LA','C',600],
                                        [21, 80,'CN','EU','B',200],
                                        [21, 80,'CN','ME','C',200],
                                        [21, 80,'CN','EU','A',300],
                                        [25, 85,'CN','EU','B',100],
                                        [30,80,'CN','EU','A',100],
                                        [35,80,'CN','EU','C',200],
                                        [36,85,'CN','ME','B',500],
                                        [40,80,'CN','LA','C',600],
                                        [51, 60,'CN','EU','B',200],
                                        [51, 80,'CN','ME','C',200],
                                        [61, 80,'CN','EU','A',300],
                                        [65, 95,'CN','EU','B',100],
                                        [70,90,'CN','EU','A',100],
                                        [75,80,'CN','EU','C',200],
                                        [76,95,'CN','ME','B',500],
                                        [80,90,'CN','LA','C',600]],
            columns = ['orderCreateDate','CRD','origin','dest','MATERIAL_CODE','quantity'])
        self.productionPlan = \
            pd.DataFrame([[0,1,'B',200,999,3,-1,-1],
                          [1,1,'C',200,999,2,-1,-1],
                          [2,1,'A',300,999,3,-1,-1]
                    ], columns = ['createDate','plannedProduction',
                         'MATERIAL_CODE','quantity','priority',
                         'RPD','customerOrder','actualProduction'])
        self.productHierarchy = \
            pd.DataFrame([['AA','A'],
                          ['AA','A1']
                          ['AA','A2'],
                          ['BB','B'],
                          ['BB','B1']
                          ['BB','B2'],
                          ['CC','C'],
                          ['DD','D']
                    ], columns = ['productGroup','MATERIAL_CODE'])
        self.BOM = pd.DataFrame([['A',None],
                                 ['A1','a1'],
                                 ['A1','a2'],
                                 ['A2','a1'],
                                 ['A2','b1'],
                                 ['B','b1'],
                                 ['B','b2'],
                                 ['B1','b1'],
                                 ['B2','b1'],
                                 ['C','c1'],
                                 ['D',None]
                    ], columns = ['FG','rawMaterial'])

def sortAgent(listOfAgents, sortKey, reverse = False):
    key = {agent.__getattribute__(sortKey): i for i, agent in enumerate(listOfAgents)}
    sequence = [key.get(x) for x in sorted(key.keys(), reverse = reverse)]
    return sequence.copy()

def calCostPerProduct(model):
    totalCost = 0
    totalNrAgents = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Product):
            if agent.active:
                totalNrAgents = totalNrAgents + 1
                totalCost = totalCost + agent.capitalCost + agent.scrapCost + \
                        agent.storageCost + agent.transportCost
    if totalNrAgents == 0:
        return 0
    else:
        return totalCost / totalNrAgents
    
def showPlantUtilization(model):
    utilization = 0
    capacity = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Plant):
            utilization = utilization + agent.averageUtilization
            capacity = capacity + agent.dailyCapacity
    if capacity == 0:
        return 0
    else:
        return utilization / capacity

def showWhsUtilization(model):
    utilization = 0
    capacity = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Warehouse):
            utilization = utilization + agent.utilization
            capacity = capacity + agent.capacity
    if capacity == 0:
        return 0
    else:
        return utilization / capacity
    
def plantUtil2(agent):
    if isinstance(agent, Plant):
        return agent.averageUtilization
    
def whsUtil2(agent):
    if isinstance(agent, Warehouse):
        return agent.utilization
    
#%%
class SupplyChainModel(Model):
    def __init__(self, width = 100, height = 100):
        self.setup = Setup()        
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(width, height, torus = False)
        self.running = True
        self.num_product = 0
        self.num_marketPlanner = 0
        self.num_transp = 0
        self.num_whs = 0
        self.num_supplier = 0
        self.num_order = 0
        self.num_plant = 0
        self.num_productionScheduler = 0
        self.rules = Rules()
        # create agents
        for key in set(self.setup.whs['REGION_EN_NAME'].tolist()):
            record = self.setup.whs[self.setup.whs['REGION_EN_NAME'] == key]
            agent = Warehouse(key, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (int(record['x']), int(record['y'])))
            self.num_whs = self.num_whs + 1
            
            market_planner = MarketPlanner('marketPlanner' + str(self.num_marketPlanner), self, agent)
            market_planner.responsibleProductGroup = 
            market_planner.responsibleRegion = key
            market_planner.marketplanDir = self.rules.marketplanDir # folder where all historical plans are
            self.schedule.add(market_planner)
            self.grid.place_agent(market_planner, (int(record['x']), int(record['y'])))
            self.num_marketPlanner = self.num_marketPlanner + 1

        tbl = self.setup.supplierCatalog.copy()
        for key in set(self.setup.supplier['supplierName'].tolist()):
            record = self.setup.supplier[self.setup.supplier['supplierName'] == key]
            listOfProducts = tbl[tbl['supplierName'] == key]
            agent = Supplier(key, self, listOfProducts)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (int(record['x']), int(record['y'])))
            self.num_supplier = self.num_supplier + 1
            
        for key in set(self.setup.plant['plantRegion'].tolist()):
            record = self.setup.plant[self.setup.plant['plantRegion'] == key]
            for i in self.schedule.agents:
                if isinstance(i, Warehouse) and i.unique_id == record['whs_name'][0]:
                    whs = i
                    break
            agent = Plant(key, self, whs)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (int(record['x']), int(record['y'])))
            self.num_plant = self.num_plant + 1
            production_scheduler = ProductionScheduler('productionScheduler' + str(self.num_productionScheduler), self, agent)
            production_scheduler.productionPlanDir = self.rules.productionPlanDir
            self.schedule.add(production_scheduler)
            self.grid.place_agent(production_scheduler, (int(record['x']), int(record['y'])))
            self.num_productionScheduler = self.num_productionScheduler + 1
        
        self.datacollector1 = DataCollector(
                model_reporters = {'cost per product': calCostPerProduct})
        self.datacollector2 = DataCollector(
                model_reporters = {'plant utilization': showPlantUtilization})
        self.datacollector3 = DataCollector(
                model_reporters = {'whs utilization': showWhsUtilization})
        self.datacollector4 = DataCollector(
                agent_reporters = {'plant util2': lambda a: plantUtil2(a)})
        self.datacollector5 = DataCollector(
                agent_reporters = {'whs util2': lambda a: whsUtil2(a)})
    
    def bookTransport(self, origin, destination, 
                      specialFreight = False, speed = 1, 
                      priceIncreaseFactor = 10, reliability = 0.9, 
                      weightCapacity = 1000, volumeCapacity = 1000, 
                      singleTripMode = True):
        unique_id = 'transport-'+str(self.num_transp)
        transport = Transportation(unique_id, self, specialFreight, speed, 
                                   priceIncreaseFactor, reliability, 
                                   weightCapacity, volumeCapacity, 
                                   singleTripMode)
        transport.origin = origin
        transport.destination = destination
        self.schedule.add(transport)
        self.grid.place_agent(transport, (origin[0], origin[1]))
        self.num_transp = self.num_transp + 1
    
    def dispatchFinishedGoods(self):

        #iterate over all plants and allocate product in plant to order at origin      
        plants = []
        orders = []
        for cell in self.grid.coord_iter():
            content, _, _ = cell
            for i in range(len(content)):
                if isinstance(list(content)[i], Plant):
                    plants.append(list(content)[i])
                if isinstance(list(content)[i], Order):
                    orders.append(list(content)[i])

        openOrders = [x for x in orders if sum(x.orderLines['quantity']) > 0]
        priority = sortAgent(openOrders, 'RPD')
        for x in plants:
            for i in priority: 
                self.allocateProduct(openOrders[i], x.pos)
                readyToShip = [prod for prod in openOrders[i].listOfProducts if prod.pos == x.pos]
                # move to commission area
                record = self.setup.originDestination[(self.setup.originDestination['origin'] == x.unique_id) & 
                                                      (self.setup.originDestination['dest'] == openOrders[i].destRegion)]
                commissionX = int(np.array(record['originX'])[0])
                commissionY = int(np.array(record['originY'])[0])
                for r in readyToShip:
                    self.grid.place_agent(r, (commissionX, commissionY))

        # iterate over all commission area (origin of origin-destination pair) to book transportation        
        origin = self.setup.originDestination.loc[:,['originX','originY']].apply(lambda row: (row[0], row[1]), axis = 1)
        destination = self.setup.originDestination.loc[:,['destX','destY']].apply(lambda row: (row[0], row[1]), axis = 1)
        for i in range(len(origin)):
            content = self.grid.get_cell_list_contents(origin[i])
            listOfProducts = [agent for agent in content if isinstance(agent, Product)]
            existingTransport = [agent for agent in content if isinstance(agent, Transportation)]
            if len(existingTransport) > 0:
                minUtilization = min([max(transp.utilizedWeight / transp.weightCapacity * 100, 
                                          transp.utilizedVolume / transp.volumeCapacity * 100) 
                                        for transp in existingTransport])
            else:
                minUtilization = 100
            if len(listOfProducts) == 0 or (minUtilization <  50):
                continue
            else:
                self.bookTransport(origin[i], destination[i])
    
    def createOrder(self, orderPerLine = True):
        orderLines = self.setup.orderLines[self.setup.orderLines['orderCreateDate'] == self.schedule.time]
        if orderLines.shape[0] == 0:
            return
        # by default: 1 order is created per orderline. 
        # there is the possibility to include multiple products (multiple order lines) into one order 
        # by omitting the 'MATERIAL_CODE' field from groupby.
        # however the order cannot yet be split in dispatchFinishedGoods or in transportation. therefore 
        # there is the risk of postponing the order when any one item cannot become available.
        if orderPerLine:
            tmp = orderLines.copy()
            for row in range(orderLines.shape[0]):
                self.dispatchOrder(pd.DataFrame([tmp.iloc[row,:]]))
        else:
            tmp = orderLines.copy().groupby(['orderCreateDate','CRD','origin','dest'], as_index = False)
            for tbl in [tmp.get_group(x) for x in tmp.groups]:
                self.dispatchOrder(tbl)

    def allocateProduct(self, orderAgent, pos):
        # allocate products to order 
        content = self.grid[int(pos[0])][int(pos[1])]
        productAtPos = [product for product in content if isinstance(product, Product)]
        productAtPos = [product for product in productAtPos if product.active and 
                        (product.bookedForCustomer is False or product.order.unique_id == orderAgent.unique_id)]
        for x in productAtPos:
            if x.name in orderAgent.orderLines['MATERIAL_CODE'].tolist():
                required = np.array(orderAgent.orderLines.loc[orderAgent.orderLines['MATERIAL_CODE'] == x.name,'quantity'])[0]
                if required == 0:
                    return
                inStorage = x.quantity
                book = min(required, inStorage)
                orderAgent.orderLines.loc[orderAgent.orderLines['MATERIAL_CODE'] == x.name, 'quantity'] \
                    = orderAgent.orderLines.loc[orderAgent.orderLines['MATERIAL_CODE'] == x.name, 'quantity'] - book
                if x.quantity == book:
                    orderAgent.listOfProducts.append(x)
                    x.order = orderAgent
                    x.bookedForCustomer = True

                else:
                    x.order = None
                    x.bookedForCustomer = False
                    xnew = x.split(book)[0]
                    
                    orderAgent.listOfProducts.append(xnew)
                    xnew.order = orderAgent
                    xnew.bookedForCustomer = True

                orderAgent.orderLines.loc[orderAgent.orderLines['MATERIAL_CODE'] == x,'quantity'] = \
                    required - book
    
    def dispatchOrder(self, orderLines, buffer = 7):
        order = Order('order' + str(self.num_order), self, orderLines)
        self.num_order = self.num_order + 1
        CRD = min(orderLines['CRD'])
        self.schedule.add(order)
        originRegion = order.originRegion
        destRegion = order.destRegion
        origin = self.setup.whs[self.setup.whs['REGION_EN_NAME'] == originRegion]
        dest = self.setup.whs[self.setup.whs['REGION_EN_NAME'] == destRegion]
        # see if order can be directly fulfilled at destination 
        self.allocateProduct(order, (int(dest['x']), int(dest['y'])))
        if sum(order.orderLines['quantity']) == 0:
            RPD = CRD
        else:
            # calculate lead time
            transportLT = np.sqrt( np.power(int(origin['x']) - int(dest['x']),2) + np.power(int(origin['y']) - int(dest['y']),2))
            RPD = CRD - int(transportLT) - buffer
            if RPD < self.schedule.time:
                RPD = self.schedule.time
                order.specialFreight = True
        # place order at origin warehouse
        order.RPD = RPD
        self.grid.place_agent(order, (int(origin['x']), int(origin['y'])))          
        
    def step(self):
        self.datacollector1.collect(self)
        self.datacollector2.collect(self)
        self.datacollector3.collect(self)
        self.datacollector4.collect(self)
        self.datacollector5.collect(self)
        self.createOrder()
        self.dispatchFinishedGoods()
        self.schedule.step()
            
#%%

model = SupplyChainModel()

for i in range(200):
    model.step()
agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    content, x, y = cell
    agent_counts[x][y] = len(content)
pd.DataFrame(agent_counts).to_clipboard()

data1 = model.datacollector1.get_model_vars_dataframe()
data2 = model.datacollector2.get_model_vars_dataframe()
data3 = model.datacollector3.get_model_vars_dataframe()
data4 = model.datacollector4.get_agent_vars_dataframe()
data5 = model.datacollector5.get_agent_vars_dataframe()
data4 = data4[~data4['plant util2'].isnull()]
data5 = data5[~data5['whs util2'].isnull()]
