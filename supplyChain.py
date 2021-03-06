# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 13:38:59 2017

@author: Nat
"""
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
import random
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import os
os.chdir('D:\\projects\\mesa')
import warnings
warnings.filterwarnings('ignore')
import re

class Transportation(Agent):
    def __init__(self, unique_id, model, specialFreight = False, speed = 1, 
                 priceIncreaseFactor = 10, reliability = 0.9, 
                 weightCapacity = 10000, volumeCapacity = 10000, singleTripMode = True):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.weightCapacity = weightCapacity # as absolute units
        self.volumeCapacity = volumeCapacity # as absolute units
        self.utilizedWeight = 0  # as absolute units
        self.utilizedVolume = 0  # as absolute units
        self.reliability = reliability # e.g. 0.9 is 90%
        self.unitWeightPrice = speed / 100 # price per weight unit per distance unit, related to speed / transport mode
        self.unitVolumePrice = speed / 100
        self.priceIncreaseFactor = priceIncreaseFactor
        self.volumeDiscount = {0:0, 50:0.1, 80:0.15, 95:0.2} # if utilized capacity > percentage, then price is discounted by decimal
        self.minUtilizationToSail = self.model.rules.minTranspUtilization # minimum utiliation percentage to sail, e.g. 50 is 50%
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
        
    def advance(self):
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
        for i in range(self.speed):
            if self.pos == self.destination:
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
        totalDistance = self.calDistance([self.origin])[0]
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
        tmpUnitWeightPrice = tmpUnitWeightPrice * (1 - discount) * totalDistance
        tmpUnitVolumePrice = tmpUnitVolumePrice * (1 - discount) * totalDistance
        
        for product in self.listOfProducts:
            product.transportCost += max(product.weight * product.quantity * tmpUnitWeightPrice, 
                                        product.volume * product.quantity * tmpUnitVolumePrice)
            product.transportCostForAllocation += product.transportCost
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
        self.value = 1 # unit value
        self.supplier = supplier
        self.decayRate = 0.001 # absolute value every time unit to subtract from shelfLifeLeft. 
                              # when down to zero the product should be scrapped
        # supply chain characteristics
        self.order = None
        self.deliveredQuantity = quantity # quantity originally delivered with the order
        self.quantity = quantity # actual quantity (e.g. after consumption or after order split)
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
        self.purchaseCost = 0 # calculated by supplier
        self.transportCost = 0 # calculated by spediteur
        self.capitalCost = 0
        self.storageCost = 0 # calculated by warehouse
        self.scrapCost = 0
        # dynamic costs in case of order split or consumption
        self.purchaseCostForAllocation = 0 # proportional to original cost if material is consumed
        self.transportCostForAllocation = 0 # proportional to original cost if material is consumed

    def step(self):
        if not self.active:
            return
        self.shelfLifeLeft = self.shelfLifeLeft - self.decayRate
    
    def advance(self):
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
        product.purchaseCostForAllocation = self.purchaseCostForAllocation * partialQuantity / self.quantity
        product.transportCostForAllocation = self.transportCostForAllocation * partialQuantity / self.quantity
        
        self.quantity = self.quantity - partialQuantity
        self.deliveredQuantity = self.quantity
        self.purchaseCost = self.purchaseCost - product.purchaseCost
        self.transportCost = self.transportCost - product.transportCost
        self.capitalCost = self.capitalCost - product.capitalCost
        self.storageCost = self.storageCost - product.storageCost
        self.scrapCost = self.scrapCost - product.scrapCost
        self.purchaseCostForAllocation = self.purchaseCostForAllocation - product.purchaseCostForAllocation
        self.transportCostForAllocation = self.transportCostForAllocation - product.transportCostForAllocation
        
        self.model.schedule.add(product)
        self.model.num_product = self.model.num_product + 1
        self.model.grid.place_agent(product, self.pos)
        
        return [self, product]
        
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
        
    def advance(self):
        self.calStorageCost()
    
    def calStorageCost(self):
        for product in self.listOfProducts:
            if product.active:
                product.storageCost = self.unitPrice / self.stackLayer * \
                                      product.volume * product.quantity
                self.totalStorageCost = self.totalStorageCost + product.storageCost        
#%%    
class Supplier(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.name = None
        self.pos = (0, 0)
        self.productCatalog = None
        self.orders = None
        self.reliability = 0.9
        self.utilization = 0
        self.dailyCapacity = 10000000
    
    def calPurchaseCost(self, productName, quantity):
        catalog = self.productCatalog[self.productCatalog['MATERIAL_CODE'] == productName]
        discountLevel = np.array(catalog['volumeDiscountCode'])
        idx = np.sum(discountLevel <= quantity) - 1
        applicablePrice = np.array(catalog.loc[catalog['volumeDiscountCode'] == discountLevel[idx],'unitPrice'])[0]
        return applicablePrice * quantity
    
    def produce(self):
        # TODO: add capacity constraint
        if self.orders is None:
            return
        self.orders.index = np.arange(0, self.orders.shape[0])
        openOrder = self.orders[(self.orders['RPD'] <= self.model.schedule.time) & 
                                (self.orders['actualProductionDate'] == -1)]
        openOrder.sort_values(by = 'RPD', ascending = True, inplace = True)
        if openOrder.shape[0] == 0:
            return
        for i in range(openOrder.shape[0]):
            # simulate reliability 
            if random.randrange(0, 100) > self.reliability * 100:
                continue

            line = pd.DataFrame([openOrder.iloc[i, :]])
            idx = line.index
            quantity = np.array(line.loc[:,'quantity'])[0]
            orderID = np.array(line.loc[:,'orderID'])[0]

            if self.utilization + quantity > self.dailyCapacity:
                # TODO: add split-order rules
                continue

            productName = np.array(line.loc[:, 'MATERIAL_CODE'])[0]

            record = self.model.setup.product[self.model.setup.product['MATERIAL_CODE'] == productName]
            if record.shape[0] > 0:
                wgt = int(np.array(record.weight)[0])
                vol = int(np.array(record.volume)[0])
            else:
                wgt = self.model.rules.default_weight # default
                vol = self.model.rules.default_volume # default

            product = Product('material' + str(self.model.num_product), 
                              self.model, 
                              quantity = quantity,
                              supplier = self.unique_id, 
                              weight = wgt, 
                              volume = vol)
            product.name = productName
            product.purchaseCost += self.calPurchaseCost(product.name, product.quantity) # total purchase cost of the batch
            product.purchaseCostForAllocation = product.purchaseCost
            # TODO: add consideration of incoterms and its influence on price / costing
            product.value = product.purchaseCost / quantity # unit value of the product
            
            if orderID is not None:
                content = self.model.grid.get_cell_list_contents(self.pos)
                for order in content:
                    if isinstance(order, Order) and order.unique_id == orderID:
                        product.order = order
                        product.bookedForCustomer = True
                        product.priority = product.order.RPD
                        break       
            
            self.model.schedule.add(product)
            self.model.grid.place_agent(product, self.pos)
            self.model.num_product = self.model.num_product + 1
            self.utilization = self.utilization + quantity
            self.orders.loc[idx, 'actualProductionDate'] = self.model.schedule.time
    
    def manageVMI(self):
        #TODO: add vendor managed inventory
        return         
    
    def readInOrders(self):
        orders = [x for x in self.model.grid[self.pos[0]][self.pos[1]] if isinstance(x, Order)]
        openOrders = [x for x in orders if sum(x.orderLines['quantity']) > 0]
        currentOpenOrders = [x for x in openOrders if x.creationDate == self.model.schedule.time]
        if len(currentOpenOrders) == 0:
            return
        
        priority = sortAgent(currentOpenOrders, 'RPD', reverse = True) # least important first to be scheduled in backwards scheduling
        
        demand = []
        for i in priority:
            record = currentOpenOrders[i].orderLines.loc[:, ['MATERIAL_CODE','quantity']]
            record['RPD'] = currentOpenOrders[i].RPD
            record['orderID'] = currentOpenOrders[i].unique_id
            if len(demand) == 0:
                demand = record
            else:
                demand = pd.concat([demand, record], axis = 0)
        demand['actualProductionDate'] = -1        
        self.orders = demand.copy()
    
    def step(self):
        self.readInOrders()
        self.manageVMI()
        self.produce()
    
    def advance(self):
        self.utilization = 0
        return

class MaterialPlanner(Agent):
    def __init__(self, unique_id, model, plant = None):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.responsibleRawMaterial = list()
        self.reliability = 0.9
        self.demand = None
        self.currentMaterialPlan = None
        self.plant = plant
        self.kpi = None
        self.planningHorizon = 14
        self.orderLines = None
    
    def BOMexplosion(self):
        productionPlan = self.plant.productionPlan[self.plant.productionPlan['actualProduction'] == -1]
        if productionPlan.shape[0] == 0:
            return
        BOM = self.model.setup.BOM.copy()
        BOM = BOM[~BOM['rawMaterial'].isnull()]
        coln = BOM.columns.tolist()
        coln[coln.index('FG')] = 'MATERIAL_CODE'
        coln[coln.index('quantity')] = 'RMquantity'
        BOM.columns = coln
        
        df = productionPlan.merge(BOM, how = 'inner', on = 'MATERIAL_CODE')
        df['RMrequiredQty'] = df['RMquantity'] * df['quantity']
        df = df.loc[:,['plannedProduction', 'rawMaterial', 'RMrequiredQty']].groupby(
                ['plannedProduction', 'rawMaterial'], as_index = False).sum()
        df.sort_values('plannedProduction', inplace = True)
        return df
        
    def updateMaterialPlan(self):
        materialBOM = self.BOMexplosion()
        respMaterialBOM = materialBOM[materialBOM['rawMaterial'].isin(self.responsibleRawMaterial)]

        # match material in stock with planned production date and quantity
        mat = list(set(respMaterialBOM['rawMaterial']))
        content = self.model.grid[self.pos[0]][self.pos[1]]
        rawMaterial = [x for x in content if isinstance(x, Product) and x.name in mat]
        names = [x.name for x in rawMaterial]
        qty = [x.quantity for x in rawMaterial]
        df = pd.DataFrame({'rawMaterial': names, 'availableQty': qty})       
        df = df.groupby('rawMaterial', as_index = False).sum()

        for i in range(respMaterialBOM.shape[0]):
            if df.shape[0] == 0:
                break
            material = respMaterialBOM.iloc[i, respMaterialBOM.columns.tolist().index('rawMaterial')]
            required = respMaterialBOM.iloc[i, respMaterialBOM.columns.tolist().index('RMrequiredQty')]
            available = np.array(df.loc[df['rawMaterial'] == material, 'availableQty'])
            if len(available) == 0:
                continue
            respMaterialBOM.iloc[i, respMaterialBOM.columns.tolist().index(
                                                'RMrequiredQty')] = required - np.min([required, available])
            df.loc[df['rawMaterial'] == material, 'availableQty'] = available - np.min([required, available])
            if np.sum(df['availableQty']) == 0:
                break
        respMaterialBOM = respMaterialBOM[respMaterialBOM['RMrequiredQty'] > 0]
        if respMaterialBOM.shape[0] == 0:
            return
        
        # match stock-in-transit with planned production date and quantity
        inTransit = pd.DataFrame([], columns = ['rawMaterial', 'availableQty', 'CRD'])
        tup = self.model.setup.supplier.apply(lambda row: (row.x, row.y),axis = 1).tolist()
        content = self.model.grid.iter_cell_list_contents(tup)
        matOrders = [x for x in content if isinstance(x, Order) and x.fulfilled is False]
        for openOrder in matOrders:
            for prod in openOrder.listOfProducts:
                if not prod.pos == self.pos:
                    tmp = pd.DataFrame([[prod.name, prod.quantity, openOrder.CRD]], 
                                       columns = ['rawMaterial', 'availableQty', 'CRD'], 
                                       index = [inTransit.shape[0] + 1])
                    inTransit = pd.concat([inTransit, tmp], axis = 0)
        
        inTransit = inTransit.groupby(['rawMaterial','CRD'], as_index = False).sum()
        try:
            inTransit.sort_values('CRD', inplace = True)
        except KeyError:
            inTransit = pd.DataFrame([], columns = ['rawMaterial', 'availableQty', 'CRD'])
        for i in range(respMaterialBOM.shape[0]):
            if inTransit.shape[0] == 0:
                break      
            plannedDate = respMaterialBOM.iloc[i, respMaterialBOM.columns.tolist().index('plannedProduction')]
            material = respMaterialBOM.iloc[i, respMaterialBOM.columns.tolist().index('rawMaterial')]   
            try:
                earliestArrival = np.array(inTransit.loc[inTransit['rawMaterial'] == material, 'CRD'])[0]
            except IndexError:
                earliestArrival = self.model.schedule.time + self.planningHorizon + 1
            # available quantity will not be offset if planned arrival is later than planned production date
            if plannedDate < earliestArrival:
                continue
            required = respMaterialBOM.iloc[i, respMaterialBOM.columns.tolist().index('RMrequiredQty')]
            available = inTransit.loc[(inTransit['rawMaterial'] == material) & 
                                      (inTransit['CRD'] <= plannedDate), 'availableQty']
            respMaterialBOM.iloc[i, 
                respMaterialBOM.columns.tolist().index('RMrequiredQty')] = required - np.min([required, np.sum(available)])
            for j in available.index:
                inTransit.loc[inTransit.index == j, 'availableQty'] = available[j] - np.min([required, available[j]])
                required = required - np.min([required, available[j]])
            
            inTransit = inTransit[inTransit['availableQty'] > 0]
            if inTransit.shape[0] == 0:
                break               
        respMaterialBOM = respMaterialBOM[respMaterialBOM['RMrequiredQty'] > 0]
        if respMaterialBOM.shape[0] == 0:
            return        

        coln= respMaterialBOM.columns.tolist()
        coln[coln.index('plannedProduction')] = 'CRD'
        coln[coln.index('rawMaterial')] = 'MATERIAL_CODE'
        coln[coln.index('RMrequiredQty')] = 'quantity'
        respMaterialBOM.columns = coln 
        self.demand = respMaterialBOM.copy()
        
    def placeOrder(self):
        orderLines = self.demand.copy()
        
        # determine supplier and order quantity
        # TODO: add choice of supplier based on e.g. price or reliability / quality
        # TODO: add ordering strategy e.g. MOQ
        supplierCatalog = self.model.setup.supplierCatalog.copy()
        catalog = supplierCatalog.loc[supplierCatalog['MATERIAL_CODE'].isin(orderLines['MATERIAL_CODE']),
                                      ['MATERIAL_CODE','supplierName']]
        catalog.drop_duplicates(inplace = True)
        check = catalog.groupby('MATERIAL_CODE').count()
        multipleSource = orderLines.loc[orderLines['MATERIAL_CODE'].isin(check[check['supplierName'] > 1].index),:]
        singleSource = orderLines[~orderLines.index.isin(multipleSource.index)]
        singleSource = singleSource.merge(catalog, how = 'left', on = 'MATERIAL_CODE')
        singleSource['origin'] = singleSource['supplierName']
        singleSource.drop('supplierName', axis = 1, inplace = True)
            
        try:
            multipleSource = self.selectSupplierByCost(multipleSource)
        except KeyError:
            multipleSource = pd.DataFrame([], columns = singleSource.columns)

        orderLines = pd.concat([singleSource, multipleSource], axis = 0)
        # add columns to comply with order format in Setup.orderLines
        requiredCol = self.model.setup.orderLines.columns.tolist()
        orderLines['dest'] = self.plant.unique_id
        # orders will be added to Setup.orderLines and will be created by 
        # SupplyChainModel.createOrder() in the next model.schedule.step()
        orderLines['orderCreateDate'] = self.model.schedule.time + 1
        orderLines = orderLines.loc[:, requiredCol]
        
        # create orders within planning horizon. rest is forecast and only stored in MaterialPlanner.orderLines
        if self.orderLines is None:
            previousOrders = pd.DataFrame([], columns = orderLines.columns)
        else:
            previousOrders = self.orderLines.loc[self.orderLines['CRD'] <= self.model.schedule.time + self.planningHorizon]
        self.orderLines = orderLines.copy()
        
        # only pass delta to the supplier.
        # the last day within planning horizon (model.schedule.time + MaterialPlanner.planningHorizon + 1) is full order
        coln= previousOrders.columns.tolist()
        coln[coln.index('quantity')] = 'oldQty'
        previousOrders.columns = coln
        previousOrders.drop('orderCreateDate', axis = 1, inplace = True)
        orderLines.drop('orderCreateDate', axis = 1, inplace = True)
        groupCol = orderLines.columns.tolist()
        groupCol.remove('quantity')
        
        delta = orderLines.merge(previousOrders, on = groupCol, how = 'outer')
        delta.fillna(0, inplace = True)
        # delta is one day longer than previousOrders, therefore the last day of planningHorizon is with 
        # full order quantity from the new orderLines.
        delta = delta[delta['CRD'] <= self.model.schedule.time + self.planningHorizon + 1]
        # delete consumed amount in the day's production
        presentDay = delta[delta['CRD'] == self.model.schedule.time]
        future = delta[delta['CRD'] > self.model.schedule.time]
        future['quantity'] = future['quantity'] - future['oldQty']
        delta = pd.concat([presentDay, future], axis = 0)
        delta.drop('oldQty', axis = 1, inplace = True)
        
        # check if there are changes within frozen period
        # TODO: decide if to use special freight when there is demand increase within frozen period
        reducedDemand = delta.loc[(delta['CRD'] <= self.model.schedule.time + self.planningHorizon) &
                                       (delta['quantity'] < 0), :]
        delta = delta[~delta.index.isin(reducedDemand.index)]
        reducedDemand['CRD'] = self.model.schedule.time + self.planningHorizon + 1
        delta = pd.concat([delta, reducedDemand], axis = 0)
        delta['orderCreateDate'] = self.model.schedule.time + 1
        delta = delta[delta['quantity'] != 0]
        delta['type'] = 'materialOrder'
        
        self.model.setup.orderLines = pd.concat([self.model.setup.orderLines, delta], axis = 0)
    
    def selectSupplierByCost(self, _orderLines, minObs = 5):
        groupCol = _orderLines.columns.tolist()
        groupCol.remove('quantity')
        orderLines = _orderLines.groupby(groupCol).sum()
        material = list(set(orderLines['MATERIAL_CODE']))
        supplierCatalog = self.model.setup.supplier
        result = []
        
        for i in range(len(material)):
            mat = [x for x in self.model.schedule.agents if 
                   isinstance(x, Product) and 
                   x.name == material[i] and 
                   x.supplier is not None and 
                   x.purchaseCost > 0 and
                   x.transportCost > 0]
            tmp = orderLines[orderLines['MATERIAL_CODE'] == material[i]]
            # check if there are new suppliers 
            availableSupplier = set(supplierCatalog.loc[supplierCatalog['MATERIAL_CODE'] == material[i], 'supplierName'])
            sup = [x.supplier for x in mat]
            counter = Counter(sup)
            # if there are new suppliers which have no previous delivery, or if minimum observation point (number of 
            # orders per supplier) is smaller than minObs, then randomly select one supplier from available suppliers
            if min(counter.values()) < minObs or len(availableSupplier - set(sup)) > 0:
                availableSupplier = list(availableSupplier)
                tmp['origin'] = tmp.apply(lambda row: availableSupplier[random.randint(0, len(availableSupplier) - 1)], axis = 1)
                if len(result) == 0:
                    result = tmp.copy()
                else:
                    result = pd.concat([result, tmp], axis = 1)
                continue
            
            # if there are enough observation points to do regression
            availableSupplier = list(availableSupplier)
            crd = [x.order.CRD for x in mat]
            crd = list(map(int, crd))
            qty = [x.deliveredQuantity for x in mat]
            cost = [x.purchaseCost + x.transportCost for x in mat]
            
            hist = pd.DataFrame({'CRD': crd,'quantity': qty,'supplierName': sup,'cost': cost})
            hist = pd.get_dummies(hist)
            fcst = pd.DataFrame([], columns = hist.columns)
            for j in range(len(availableSupplier)):
                tmp0 = tmp.copy()
                tmp0['supplierName'] = availableSupplier[j]
                tmp0['cost'] = 0
                tmp0 = pd.get_dummies(tmp0)
                tmp0 = tmp0.loc[:, hist.columns]
                fcst = pd.concat([fcst, tmp0], axis = 0)
            fcst.fillna(0, inplace = True)
            
            # regress to find cost for specific suppliers
            coln = hist.columns.tolist()
            coln.remove('cost')
            coln.remove('CRD')
            lrmodel = lr()
            modelfit = lrmodel.fit(hist.loc[:, coln], hist.loc[:, 'cost'])
            pred = modelfit.predict(fcst.loc[:, coln])
            fcst['cost'] = pred
            # sort by lowest cost first, and return supplier with lowest cost
            fcst.sort_values(['CRD','cost'], inplace = True)
            fcst = fcst.groupby(['CRD', 'quantity'], as_index = False).first()
            # get supplier name from one-hot dummies
            pos = fcst.iloc[:, -len(availableSupplier):]
            fcst['origin'] = pos.apply(lambda row: availableSupplier[row.tolist().index(1)], axis= 1)
            tmp = tmp.merge(fcst.loc[:, ['CRD','quantity', 'origin']], 
                            how = 'left', on = ['CRD','quantity'])
            
            if len(result) == 0:
                result = tmp.copy()
            else:
                result = pd.concat([result, tmp], axis = 1)
            
        return result
    
    def step(self):
        return
    
    def advance(self):
        self.productionPlan = self.plant.productionPlan.copy()
        self.updateMaterialPlan()
        self.placeOrder()
    
#%%
class MarketPlanner(Agent):
    def __init__(self, unique_id, model, whs = None):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.responsibleProductGroup = list()
        self.responsibleRegion = list()
        self.reliability = 0.9
        self.marketplanDir = None # folder where all historical plans are
        self.currentMarketPlan = None
        self.whs = whs
        self.kpi = None
        self.planningInterval = 15
        self.planningHorizon = 25 # minumum planning horizon, within which the market plan is not considered
    
    def step(self):
        # TODO: schedule market planning to once per month
        if np.mod(self.model.schedule.time, self.planningInterval) == 0:
            self.updatePlan()
            self.passOrder()
    
    def advance(self):
        self.kpi = self.calKpi()
    
    def updatePlan(self):
        '''
        update market plan.
        forecast algorithm can be plugged in here as one option.
        '''
        # read in previous forecast
        try:
            previousPlan = pd.read_csv(self.marketplanDir + 'marketPlan_' + 
                                             str(self.model.schedule.time - self.planningInterval) + '.csv')
        except FileNotFoundError:
            previousPlan = self.currentMarketPlan
        coln = ['originRegion','destRegion','forecastProduct','CRD','customer']
        previousPlan = previousPlan.loc[:,coln + ['quantity']]
        previousPlan.columns = coln + ['oldQuantity']
      
        # create current forecast.
        # here run whatever algorithm
        # TODO: add the whatever algorithm
        # for simulation purpose: no update of market plan
        try:
            self.currentMarketPlan = pd.read_csv(self.marketplanDir + 'marketPlan_' + 
                                                 str(self.model.schedule.time) + '.csv')
            self.currentMarketPlan['createDate'] = self.model.schedule.time
            # write to disk the complete current market plan
            self.currentMarketPlan.to_csv(self.marketplanDir + 'marketPlan_' + 
                                                 str(self.model.schedule.time) + '.csv', index = False)
            # pass on the delta market plan 
            delta = self.currentMarketPlan.merge(previousPlan, how = 'outer', on = coln)
            delta.fillna(0, inplace = True)
            delta['quantity'] = delta['quantity'] - delta['oldQuantity']
            delta.drop('oldQuantity', axis = 1, inplace = True)
            self.currentMarketPlan = delta.copy()
        except FileNotFoundError:
            self.currentMarketPlan['quantity'] = 0
       
    def calKpi(self):
        '''
        get business KPI value 
        '''
        #TODO: add kpi calculation methods
        return 'NA'
        
    def passOrder(self):        
        '''
        split order and give order to corresponding production scheduler
        planningHorizon: 
            earliest RPD date from current date to consider long term market planning. 
            any time unit before the planning horizon does not take market planning as reference, 
            but actual orders.
        '''
        currentPlan = self.currentMarketPlan[(self.currentMarketPlan['destRegion'].isin(self.responsibleRegion)) & 
                                             (self.currentMarketPlan['forecastProduct'].isin(self.responsibleProductGroup)) & 
                                             (self.currentMarketPlan['RPD'] >= self.model.schedule.time + self.planningHorizon) &
                                             (self.currentMarketPlan['quantity'] != 0)]        
        groups = currentPlan.groupby('originRegion')
       
        plants = self.model.setup.plant.copy()
        for region in set(currentPlan['originRegion']):
            x = int(plants.loc[(plants['plantRegion'] == region), 'x'])
            y = int(plants.loc[(plants['plantRegion'] == region), 'y'])
            plan = groups.get_group(region)
            plan['RPD'] = plan['RPD'].apply(int)
            content = self.model.grid[x][y]
            productionscheduler = [x for x in content if isinstance(x, ProductionScheduler)]
            productionscheduler = productionscheduler[0]
            if productionscheduler.marketPlan.shape[0] > 0:
                productionscheduler.marketPlan = pd.concat([productionscheduler.marketPlan,plan], axis = 0)
            else:
                productionscheduler.marketPlan = plan

class Order(Agent):
    def __init__(self, unique_id, model, orderLines):
        super().__init__(unique_id, model)
        self.name = 'customerOrder'
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
            
    def advance(self):
        return
    
#%%
class ProductionScheduler(Agent):
    def __init__(self, unique_id, model, plantAgent):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.productionPlan = pd.DataFrame([]) # final production plan as passed on to plant production
        self.plant = plantAgent
        self.marketPlan = pd.DataFrame([]) # original as per market planner
        self.productionPlanDir = None
        self.reliability = 0.9
        self.capacityRedLight = list()
        self.fulfillmentRedLight = list()
        self.minProdBatch = 100
        self.availCap = 0.8 # capacity upper limit for planning
        self.scheduleNonAdherence = 0 # number of planned production orders which are not executed as planned
    
    def readInOrders(self):
        orders = [x for x in self.model.grid[self.pos[0]][self.pos[1]] if isinstance(x, Order)]
        openOrders = [x for x in orders if sum(x.orderLines['quantity']) > 0]
        currentOpenOrders = [x for x in openOrders if x.creationDate == self.model.schedule.time]
        if len(currentOpenOrders) == 0:
            return
        
        priority = sortAgent(currentOpenOrders, 'RPD', reverse = True) # least important first to be scheduled in backwards scheduling
        plan = self.productionPlan.copy()
        for i in priority: 
            line = pd.DataFrame([], columns = self.productionPlan.columns.tolist())
            for row in range(currentOpenOrders[i].orderLines.shape[0]):
                record = currentOpenOrders[i].orderLines.iloc[row, :]
                line['MATERIAL_CODE'] = [record['MATERIAL_CODE']]
                line['createDate'] = [self.model.schedule.time]
                line['RPD'] = [currentOpenOrders[i].RPD]
                line['priority'] = [999]               
                line['customerOrder'] = [currentOpenOrders[i].unique_id]
                line['actualProduction'] = [-1]
                line['quantity'] = [int(record['quantity'] / self.minProdBatch) * self.minProdBatch]
                line['plannedProduction'] = [self.model.schedule.time]
                
                # backwards planning. check if the day is already overcapacity. if so, go backwards one more day for more capacity. 
                if currentOpenOrders[i].RPD <= self.model.schedule.time:
                    # TODO: warning system and corresponding mitigation to be defined.
                    self.fulfillmentRedLight.append(currentOpenOrders[i].unique_id)
                else:
                    for i in np.arange(1, currentOpenOrders[i].RPD - self.model.schedule.time):
                        # backwards planning to search for capacity until current time unit.
                        if (sum(plan.loc[plan['plannedProduction'] == currentOpenOrders[i].RPD - i,
                                 'quantity']) + np.array(line['quantity'])[0]) > self.plant.dailyCapacity:
                            continue
                        else: 
                            line['plannedProduction'] = [currentOpenOrders[i].RPD - i]
                            break

                plan = pd.concat([plan, line])
                if sum(plan.loc[plan['plannedProduction'] == self.model.schedule.time,
                                'quantity']) > self.plant.dailyCapacity:
                    # TODO: warning system and corresponding mitigation to be defined.        
                    self.capacityRedLight.append(self.model.schedule.time)
        self.productionPlan = plan.copy()
        
    def getFinishedGoodsRatio(self, cutoffDays = 120):
        group = self.model.setup.orderLines[self.model.schedule.time - 
                            self.model.setup.orderLines['orderCreateDate'] <= cutoffDays].groupby(
                            'MATERIAL_CODE', as_index = False).sum()
        matchingTbl = self.model.setup.productHierarchy
        group = group.merge(matchingTbl, on = 'MATERIAL_CODE', how = 'left')
        bigGroup = group.groupby('forecastProduct', as_index = False).sum()
        bigGroup = bigGroup.loc[:, ['forecastProduct', 'quantity']]
        bigGroup.columns = ['forecastProduct', 'totalQuantity']
        group = group.merge(bigGroup, on = 'forecastProduct', how = 'left')
        group['ratio'] = group.apply(lambda row: row['quantity'] / row['totalQuantity'], axis = 1)
        return group.loc[:, ['MATERIAL_CODE','ratio']]
    
    def splitMarketPlan(self, cutoffDays = 120, planUnit = 'monthly'):
        '''
        split market plans to production quantity per time unit, same format as production plan
        available capacity: 
            daily capacity upper limit to plan for long-term market forecast. 
            0.8 means 80% of self.plant.dailyCapacity will be planned for market forecast.
        cutoffDays:
            number of days in history to take as reference for detailed MATERIAL_CODE level
            quantity split.
        output: new production plan with market forecast included
        '''
        if self.marketPlan.shape[0] == 0:
            return
        # TODO: add other market plan split rules
        planningInterval = 30 # default is monthly 
        if planUnit == 'weekly':
            planningInterval = 7
        if planUnit == 'seasonal':
            planningInterval = 90
        if planUnit == 'annual':
            planningInterval = 365
            
        ratio = self.getFinishedGoodsRatio(cutoffDays)
        group = self.marketPlan.groupby(['RPD','forecastProduct'], as_index = False).sum()
        colnames = group.columns.tolist()
        colnames[colnames.index('quantity')] = 'totalQuantity'
        group.columns = colnames
        
        matchingTbl = self.model.setup.productHierarchy.copy()
        group = group.merge(matchingTbl, how = 'left', on = 'forecastProduct')
        group = group.merge(ratio, how = 'left', on = 'MATERIAL_CODE')
        group['quantity'] = group['totalQuantity'] * group['ratio']
        group['quantity'] = group['quantity'].apply(int)
        
        group = group.loc[:, ['RPD','MATERIAL_CODE', 'quantity']]
        dailyCap = self.plant.dailyCapacity * self.availCap

        plan = self.productionPlan.copy()
        for dueDate in set(group['RPD']):
            rpdGroup = group[group['RPD'] == dueDate]
            totalPlanned = np.sum(rpdGroup['quantity'])
            nrPeriods = int(np.ceil(totalPlanned / dailyCap))
            if nrPeriods > planningInterval: 
                #TODO: warning system and corresponding mitigation to be defined.
                self.capacityRedLight.append(dueDate)
            
            for prod in rpdGroup['MATERIAL_CODE'].tolist():
                quantityToBePlanned = np.array(rpdGroup.loc[rpdGroup['MATERIAL_CODE'] == prod, 'quantity'])[0]
                quantityLeft = quantityToBePlanned
                
                if quantityLeft < 0:                    
                    # reduced forecast: delta passed from market planner is negative
                    _quantityLeft = int(np.abs(quantityLeft) / self.minProdBatch) * self.minProdBatch
                    for j in range(dueDate - self.model.schedule.time - 1):
                        if _quantityLeft <= 0:
                            break
                        record = plan.loc[(plan['MATERIAL_CODE'] == prod) &
                                          (plan['plannedProduction'] == dueDate - j) & 
                                          (plan['customerOrder'] == -1) & 
                                          (plan['actualProduction'] == -1),:]
                        if record.shape[0] == 0:
                            continue
                        
                        for k in range(record.shape[0]):
                            qty = record.iloc[k, record.columns.tolist().index('quantity')]
                            record.iloc[k, record.columns.tolist().index('quantity')] = np.max([0, qty - _quantityLeft])
                            _quantityLeft = _quantityLeft - qty
                            if _quantityLeft <= 0:
                                break
                        plan.loc[plan.index.isin(record.index),'quantity'] = record['quantity']
                        plan.loc[plan.index.isin(record.index),'actualProduction'] = -99
                            
                for i in range(dueDate - self.model.schedule.time - 1):
                    # check if all required quantity for the specific material and RPD are planned 
                    # because of the round-up to minimum production batch, quantity may be all planned in less than nrPeriods.
                    if quantityLeft <= 0:
                        break
                    line = pd.DataFrame([], columns = self.productionPlan.columns.tolist())
                    line['MATERIAL_CODE'] = [prod]
                    line['createDate'] = [self.model.schedule.time]
                    line['plannedProduction'] = [dueDate - i - 1]
                    line['RPD'] = [dueDate - i]
                    line['priority'] = [999]               
                    line['customerOrder'] = [-1]
                    line['actualProduction'] = [-1]
                    qty = int(np.ceil(quantityToBePlanned / planningInterval / self.minProdBatch)) * self.minProdBatch
                    if quantityLeft <  qty - self.minProdBatch:
                        line['quantity'] = [int(np.ceil(quantityLeft / self.minProdBatch)) * self.minProdBatch]
                    else:
                        line['quantity'] = [qty]
                    
                    # backwards planning. check if the day is already overcapacity. if so, go backwards one more day for more capacity. 
                    if (sum(plan.loc[plan['plannedProduction'] == dueDate - i - 1,'quantity']) + np.array(line['quantity'])[0]) >= self.plant.dailyCapacity:
                        # if backwards planning has planned more than 30 days back:
                        if i >= 30:
                            self.capacityRedLight.append(dueDate - i)
                        continue
                    
                    quantityLeft = quantityLeft - np.array(line['quantity'])[0]
                    plan = pd.concat([plan, line])
                    
                # if until current time unit there is still no capacity:
                if quantityLeft > 0:
                    self.fulfillmentRedLight.append((prod, dueDate, quantityLeft))
                    
        self.productionPlan = plan.copy()
    
    def updateProductionPlan(self):
        # run whatever algorithms to level-schedule production plan
        # TODO: add other strategies for production scheduling        
        self.productionPlan.index = np.arange(0, self.productionPlan.shape[0])
        self.productionPlan = self.productionPlan[self.productionPlan['quantity'] != 0]
        self.productionPlan.to_csv(self.productionPlanDir + self.unique_id + '_' + 
                                             str(self.model.schedule.time) + '.csv', index = False)
        self.plant.productionPlan = self.productionPlan.copy()
    
    def monitorExecution(self):
        '''
        monitor plant production situation in the current time unit (based on previous time unit's production plan)
        especially for the 'actualProduction' flag
        '''
        self.productionPlan = self.plant.productionPlan.copy()
        delayed = self.productionPlan[(self.productionPlan['plannedProduction'] <= self.model.schedule.time) &
                                        (self.productionPlan['actualProduction'] == -1)].shape[0]
        # re-schedule delayed orders from the previous time unit
        self.productionPlan.loc[(self.productionPlan['plannedProduction'] < self.model.schedule.time) &
                                        (self.productionPlan['actualProduction'] == -1), 
                                'plannedProduction'] = self.model.schedule.time + 1
        self.scheduleNonAdherence = self.scheduleNonAdherence + delayed
        
    def step(self):
        return
        
    def advance(self):
        # all market planners pass orders to production scheduler in MarketPlanner.step() methods in the current time unit.
        # based on which the production schedule will be updated in ProductionScheduler.advance() method in the current time unit
        # and carried out in Plant.step() method in the next time unit.
        self.monitorExecution()
        self.splitMarketPlan() 
        self.readInOrders()
        self.updateProductionPlan()
        self.marketPlan = pd.DataFrame([], columns = self.marketPlan.columns) # reset marketPlan to prepare for receiving new ones in next time unit
        
#%%        
class Plant(Agent):
    def __init__(self, unique_id, model, whsAgent, dailyCapacity = 2000, reliability = 0.95):
        super().__init__(unique_id, model)
        self.pos = (0, 0)
        self.whs = whsAgent
        self.dailyCapacity = dailyCapacity
        self.reliability = reliability
        self.averageUtilization = 0
        self.utilization = 0
        self.productionPlan = []
    
    def step(self):
        self.produce() # based on input from ProductionScheduler.advance() method from previous time unit.
    
    def advance(self):
        self.calUtilization()        
    
    def produce(self):
        if len(self.productionPlan) == 0:
            return
        productionPlan = self.productionPlan[(self.productionPlan['plannedProduction'] <= self.model.schedule.time) &
                                        (self.productionPlan['actualProduction'] == -1)]
        # TODO: add other production prioritization rules
        productionPlan.sort_values(by = ['priority', 'RPD'], ascending = True, inplace = True)
        for i in range(productionPlan.shape[0]):
            if random.randrange(0, 100) > self.reliability * 100:
                continue           
            idx = productionPlan.iloc[i,:].name
            line = pd.DataFrame([productionPlan.iloc[i,:]])
            productName = np.array(line['MATERIAL_CODE'])[0]
            FGquantity = int(line.loc[:,'quantity'])
            matCost = 0

            BOM = self.model.setup.BOM.copy()
            BOM = BOM[BOM['FG'] == productName]
            requiredRM = BOM['rawMaterial'].tolist()
            if not (len(requiredRM) == 0 or requiredRM[0] is None):
                content = self.model.grid[self.pos[0]][self.pos[1]]
                rawMaterial = [x for x in content if isinstance(x, Product) and x.name in requiredRM and x.availableInStorage is True]
                names = [x.name for x in rawMaterial]
                qty = [x.quantity for x in rawMaterial]
                df = pd.DataFrame({'rawMaterial': names, 'availableQty': qty})
                df = df.groupby('rawMaterial', as_index = False).sum()
                BOM['quantity'] = BOM['quantity'] * FGquantity
                df = BOM.merge(df, how = 'left', on = 'rawMaterial')
                # if any of the raw material is not available, continue to next FG production.
                if np.isnan(sum(df.availableQty)):
                    continue
                # check if partial or full availability               
                df['availableRatio'] = df.apply(lambda row: min(1, row['availableQty'] / row['quantity']), axis = 1)
                availableRatio = np.min(df['availableRatio'])
                FG_leftoverQty = FGquantity - int(FGquantity * availableRatio)
                FGquantity = FGquantity * availableRatio
                df['consumedRM'] = df['quantity'] * availableRatio
                df['consumedRM'] = df['consumedRM'].apply(int)
                df['leftoverRM'] = df['availableQty'] - df['consumedRM']
                # backflush
                for mat in rawMaterial:
                    consumedQty = int(df.loc[df['rawMaterial'] == mat.name, 'consumedRM'])
                    if mat.quantity <= consumedQty:
                        df.loc[df['rawMaterial'] == mat.name, 'consumedRM'] = consumedQty - mat.quantity
                        mat.quantity = 0
                        matCost = matCost + mat.purchaseCostForAllocation + mat.capitalCost \
                                  + mat.transportCostForAllocation + mat.storageCost
                        mat.active = False
                        mat.availableInStorage = False
                    else:
                        df.loc[df['rawMaterial'] == mat.name, 'consumedRM'] = 0
                        allocatedPurchaseCost = mat.purchaseCostForAllocation / mat.quantity * consumedQty
                        allocatedCapitalCost = mat.capitalCost / mat.quantity * consumedQty
                        allocatedTranspCost = mat.transportCostForAllocation / mat.quantity * consumedQty
                        allocatedStorageCost = mat.storageCost / mat.quantity * consumedQty
                        matCost = matCost + allocatedPurchaseCost + allocatedCapitalCost + allocatedTranspCost
                        mat.purchaseCostForAllocation = mat.purchaseCostForAllocation - allocatedPurchaseCost
                        mat.capitalCost = mat.capitalCost - allocatedCapitalCost
                        mat.transportCostForAllocation = mat.transportCostForAllocation - allocatedTranspCost
                        mat.storageCost = mat.storageCost - allocatedStorageCost
                        mat.quantity = mat.quantity - consumedQty
                # create new production schedule for leftover quantities
                adHocProduction = pd.DataFrame([np.zeros(len(self.productionPlan.columns))], 
                                                columns = self.productionPlan.columns, 
                                                index = [max(self.productionPlan.index) + 1])
                adHocProduction['createDate'] = self.model.schedule.time
                adHocProduction['plannedProduction'] = self.model.schedule.time
                adHocProduction['MATERIAL_CODE'] = productName
                adHocProduction['quantity'] = FG_leftoverQty
                adHocProduction['priority'] = np.array(line['priority'])[0]
                adHocProduction['RPD'] = np.array(line['RPD'])[0]
                adHocProduction['customerOrder'] = np.array(line['customerOrder'])[0]
                adHocProduction['actualProduction'] = -1
                self.productionPlan = pd.concat([self.productionPlan, adHocProduction], axis = 0)
                
            record = self.model.setup.product[self.model.setup.product['MATERIAL_CODE'] == productName]
            if record.shape[0] > 0:
                wgt = int(np.array(record.weight)[0])
                vol = int(np.array(record.volume)[0])
            else:
                wgt = self.model.rules.default_weight # default
                vol = self.model.rules.default_volume # default
            
            if self.utilization + FGquantity >= self.dailyCapacity:
                return

            orderID = np.array(line['customerOrder'])[0]
            product = Product(productName + str(self.model.num_product), 
                              self.model, quantity = FGquantity,
                              supplier = None, weight = wgt, volume = vol)
            product.name = productName
            
            if matCost == 0:
                productionCost = self.model.rules.defaultProductionCost
            else:
                productionCost = self.model.rules.defaultAssemblyCost
            product.value = matCost / FGquantity + productionCost
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
            self.productionPlan.loc[self.productionPlan.index == idx, 'actualProduction'] = self.model.schedule.time
    
    def calUtilization(self):
        if self.model.schedule.time > 0:
            self.averageUtilization = (self.averageUtilization * (self.model.schedule.time - 1) + self.utilization) / self.model.schedule.time
        # reset
        self.utilization = 0
        
#%%
class Rules(object):
    def __init__(self):
        self.companyInterestRate = 0.000246575342465753 #absolute value as in 9% / 365 days, assuming every time unit is one day.
        self.marketplanDir = 'D:\\projects\\mesa\\files\\'
        self.productionPlanDir = 'D:\\projects\\mesa\\files\\'
        self.default_weight = 1 # default weight per unit
        self.default_volume = 1 # default volume per unit
        self.defaultProductionCost = 3 # default cost per unit if produced in-house
        self.defaultAssemblyCost = 1 # default cost per unit if there are purchased material / semi-finished goods
        self.minTranspUtilization = 1 # 1 for 1%
        self.landTranspSpeed = 5
        self.seaTranspSpeed = 2
        self.airTranspSpeed = 20
        self.dispatchBuffer = 7 # buffer time units for order dispatch

class Setup(object):
    def __init__(self):
        self.whs = pd.DataFrame([['CN',20,60],['EU',50,10],
                                 ['ME',70,35],['LA',95,95]], 
                                columns = ['REGION_EN_NAME', 'x', 'y'])
        self.plant = pd.DataFrame([['CN',20,60,'CN',2000]],
                        columns = ['plantRegion','x','y','whs_name','capacity'])
        self.originDestination = \
            pd.DataFrame([['CN','EU',19,59,49,11,'sea'],
                          ['CN','ME',21,59,69,35,'sea'],
                          ['CN','LA',21,61,94,94,'sea'], 
                          ['supplier0','CN',29,70,21,60,'land'],
                          ['supplier1','CN',39,60,21,60,'land'],
                          ['supplier0sp','CN',30,69,21,60,'air'],
                          ['supplier1sp','CN',40,59,21,60,'air'],
                          ['CNsp','EU', 19,61,49,11,'air'],
                          ['CNsp','ME',20,59,69,35,'air'],
                          ['CNsp','LA',20,61,94,94,'air']],
               columns = ['origin','dest','originX','originY','destX','destY', 'transpMode'])
        self.product = pd.DataFrame([['A','supplier1',1,1],
                             ['B', None,1,1], ['C',None,1,1]], 
                             columns = ['MATERIAL_CODE','supplierName','weight','volume'])
        self.supplier = pd.DataFrame([['supplier0', 30,70],['supplier1',40,60]],
                                     columns = ['supplierName','x','y'])
        self.supplierCatalog = \
            pd.DataFrame([['supplier1','A',0,1],
                          ['supplier1','A',100,0.95],
                          ['supplier1','A',200,0.9],
                          ['supplier1','A',300,0.8],
                          ['supplier1','a1',0,1],
                          ['supplier1','a1',200,0.9],
                          ['supplier1','a1',500,0.8],
                          ['supplier1','a1',1000,0.7],
                          ['supplier0','a2',0,2],
                          ['supplier1','a2',0,2],
                          ['supplier1','a2',100,1.98],
                          ['supplier1','b1',0,5],
                          ['supplier0','b2',0,7],
                          ['supplier0','c1',0,2],
                          ['supplier0','c1',1000,1.8],
                          ['supplier0','c1',2000,1.6]], 
            columns = ['supplierName', 'MATERIAL_CODE','volumeDiscountCode','unitPrice'])
        self.orderLines = pd.DataFrame([[1, 60,'CN','EU','B',200,'customerOrder'],
                                        [1, 50,'CN','ME','C',200,'customerOrder'],
                                        [1, 60,'CN','EU','A1',300,'customerOrder'],
                                        [5, 65,'CN','EU','B1',100,'customerOrder'],
                                        [10,70,'CN','EU','A',100,'customerOrder'],
                                        [15,80,'CN','EU','C',200,'customerOrder'],
                                        [16,45,'CN','ME','B2',500,'customerOrder'],
                                        [20,70,'CN','LA','C',600,'customerOrder'],
                                        [21, 80,'CN','EU','B1',200,'customerOrder'],
                                        [21, 80,'CN','ME','C',200,'customerOrder'],
                                        [21, 80,'CN','EU','A',300,'customerOrder'],
                                        [25, 85,'CN','EU','B',100,'customerOrder'],
                                        [30,80,'CN','EU','A',100,'customerOrder'],
                                        [35,80,'CN','EU','C',200,'customerOrder'],
                                        [36,85,'CN','ME','B',500,'customerOrder'],
                                        [40,80,'CN','LA','C',600,'customerOrder'],
                                        [51, 60,'CN','EU','B',200,'customerOrder'],
                                        [51, 80,'CN','ME','C',200,'customerOrder'],
                                        [61, 80,'CN','EU','A2',300,'customerOrder'],
                                        [65, 95,'CN','EU','B2',100,'customerOrder'],
                                        [70,90,'CN','EU','A',100,'customerOrder'],
                                        [75,80,'CN','EU','C',200,'customerOrder'],
                                        [76,95,'CN','ME','B',500,'customerOrder'],
                                        [80,90,'CN','LA','C',600,'customerOrder']],
            columns = ['orderCreateDate','CRD','origin','dest','MATERIAL_CODE','quantity', 'type'])
        self.productionPlan = \
            pd.DataFrame([[0,1,'B',200,999,3,-1,-1],
                          [1,1,'C',200,999,2,-1,-1],
                          [2,1,'A',300,999,3,-1,-1]
                    ], columns = ['createDate','plannedProduction',
                         'MATERIAL_CODE','quantity','priority',
                         'RPD','customerOrder','actualProduction'])
        self.productHierarchy = \
            pd.DataFrame([['AA','A'],
                          ['AA','A1'],
                          ['AA','A2'],
                          ['BB','B'],
                          ['BB','B1'],
                          ['BB','B2'],
                          ['CC','C'],
                          ['DD','D']
                    ], columns = ['forecastProduct','MATERIAL_CODE'])
        self.BOM = pd.DataFrame([['A',None, None],
                                 ['A1','a1', 1],
                                 ['A1','a2', 2],
                                 ['A2','a1', 3],
                                 ['A2','b1', 2],
                                 ['B','b1', 4],
                                 ['B','b2', 2],
                                 ['B1','b1', 5],
                                 ['B2','b1', 2],
                                 ['C','c1', 10],
                                 ['D',None, None]
                    ], columns = ['FG','rawMaterial', 'quantity'])

def sortAgent(listOfAgents, sortKey, reverse = False):
    pair = {i: agent.__getattribute__(sortKey) for i, agent in enumerate(listOfAgents)}
    sequence = sorted(pair, key = pair.__getitem__, reverse = reverse)
    return sequence.copy()

def calCostPerProduct(model):
    totalCost = 0
    totalNrAgents = 0
    for agent in model.schedule.agents:
        pattern = re.compile('^material')
        if isinstance(agent, Product) and not pattern.match(agent.unique_id):
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

def averageWhsUtilization(model):
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
    
def EUWhsUtilization(model):
    util = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Warehouse) and agent.unique_id == 'EU':
            util = agent.utilization / agent.capacity
            break
    return util

def CNWhsUtilization(model):
    util = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Warehouse) and agent.unique_id == 'CN':
            util = agent.utilization / agent.capacity
            break
    return util
    
def LAWhsUtilization(model):
    util = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Warehouse) and agent.unique_id == 'LA':
            util = agent.utilization / agent.capacity
            break
    return util

def MEWhsUtilization(model):
    util = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Warehouse) and agent.unique_id == 'ME':
            util = agent.utilization / agent.capacity
            break
    return util

def EUorderFulfillment(model):
    totalFulfilled = 0
    totalOnTime = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Order) and agent.destRegion == 'EU':
            totalFulfilled = totalFulfilled + agent.fulfilled
            totalOnTime = totalOnTime + agent.onTimeInFull
    if totalFulfilled == 0: 
        return 0
    else:
        return totalOnTime / totalFulfilled        

def MEorderFulfillment(model):
    totalFulfilled = 0
    totalOnTime = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Order) and agent.destRegion == 'ME':
            totalFulfilled = totalFulfilled + agent.fulfilled
            totalOnTime = totalOnTime + agent.onTimeInFull
    if totalFulfilled == 0: 
        return 0
    else:
        return totalOnTime / totalFulfilled     

def LAorderFulfillment(model):
    totalFulfilled = 0
    totalOnTime = 0
    for agent in model.schedule.agents:
        if isinstance(agent, Order) and agent.destRegion == 'LA':
            totalFulfilled = totalFulfilled + agent.fulfilled
            totalOnTime = totalOnTime + agent.onTimeInFull
    if totalFulfilled == 0: 
        return 0
    else:
        return totalOnTime / totalFulfilled     

#%%
class SupplyChainModel(Model):
    def __init__(self, width = 100, height = 100):
        self.setup = Setup()        
        self.schedule = SimultaneousActivation(self) # requires each agent to implement both methods 'step' and 'advance'
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
        self.num_materialPlanner = 0
        self.rules = Rules()
        # set up regional warehouse and associated market planner
        for key in set(self.setup.whs['REGION_EN_NAME'].tolist()):
            record = self.setup.whs[self.setup.whs['REGION_EN_NAME'] == key]
            agent = Warehouse(key, self)
            self.schedule.add(agent)
            self.grid.place_agent(agent, (int(record['x']), int(record['y'])))
            self.num_whs = self.num_whs + 1
            
            market_planner = MarketPlanner('marketPlanner' + str(self.num_marketPlanner), self, agent)
            market_planner.responsibleProductGroup = list(set(self.setup.productHierarchy['forecastProduct'])) # default is all
            market_planner.responsibleRegion.append(key)
            market_planner.marketplanDir = self.rules.marketplanDir # folder where all historical plans are
            self.schedule.add(market_planner)
            self.grid.place_agent(market_planner, (int(record['x']), int(record['y'])))
            self.num_marketPlanner = self.num_marketPlanner + 1
        # set up suppliers
        tbl = self.setup.supplierCatalog.copy()
        for key in set(self.setup.supplier['supplierName'].tolist()):
            record = self.setup.supplier[self.setup.supplier['supplierName'] == key]
            agent = Supplier('supplier' + str(self.num_supplier), self)
            agent.name = key
            agent.productCatalog = tbl[tbl['supplierName'] == key]
            self.schedule.add(agent)
            self.grid.place_agent(agent, (int(record['x']), int(record['y'])))
            self.num_supplier = self.num_supplier + 1
        # set up plants and associated production planners and material planners
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
            # set up production scheduler
            production_scheduler = ProductionScheduler('productionScheduler' + str(self.num_productionScheduler), self, agent)
            production_scheduler.productionPlanDir = self.rules.productionPlanDir
            production_scheduler.productionPlan = pd.read_csv(production_scheduler.productionPlanDir + 
                                                              production_scheduler.unique_id + '_' + '-1.csv')
            production_scheduler.plant.productionPlan = production_scheduler.productionPlan # initial production plan
            self.schedule.add(production_scheduler)
            self.grid.place_agent(production_scheduler, (int(record['x']), int(record['y'])))
            self.num_productionScheduler = self.num_productionScheduler + 1
            # set up material planner
            material_planner = MaterialPlanner('materialPlanner' + str(self.num_materialPlanner), self, agent)
            material_planner.responsibleRawMaterial = list(set(self.setup.BOM['rawMaterial'])) # default is all raw material
            self.schedule.add(material_planner)
            self.grid.place_agent(material_planner, (int(record['x']), int(record['y'])))
            self.num_materialPlanner = self.num_materialPlanner + 1
        
        self.datacollector = DataCollector(
                model_reporters = {'cost per product': calCostPerProduct,
                                   'plant utilization': showPlantUtilization, 
                                   'average whs utilization': averageWhsUtilization,
                                   'EU warehouse': EUWhsUtilization,
                                   'CN warehouse': CNWhsUtilization,
                                   'LA warehouse': LAWhsUtilization,
                                   'ME warehouse': MEWhsUtilization,
                                   'EU order fulfillment': EUorderFulfillment,
                                   'LA order fulfillment': LAorderFulfillment,
                                   'ME order fulfillment': MEorderFulfillment})
   
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
        plants = [x for x in self.schedule.agents if isinstance(x, (Plant, Supplier))]
        orders = [x for x in self.schedule.agents if isinstance(x, Order)]

        openOrders = [x for x in orders if sum(x.orderLines['quantity']) > 0]
        if len(openOrders) == 0:
            return
        for x in plants:
            openOrders0 = [order for order in openOrders if order.pos == x.pos]
            priority = sortAgent(openOrders0, 'RPD')
#            if len(openOrders0) 
            for i in priority: 
                self.allocateProduct(openOrders0[i], x.pos)
                readyToShip = [prod for prod in openOrders0[i].listOfProducts if prod.pos == x.pos]
                # move to commission area
                record = self.setup.originDestination[(self.setup.originDestination['origin'] == x.unique_id) & 
                                                      (self.setup.originDestination['dest'] == openOrders0[i].destRegion)]
                commissionX = int(np.array(record['originX'])[0])
                commissionY = int(np.array(record['originY'])[0])
                for r in readyToShip:
                    self.grid.place_agent(r, (commissionX, commissionY))
        
        # add special freight
        urgentOrders = [x for x in openOrders if x.specialFreight is True or x.RPD + self.rules.dispatchBuffer < self.schedule.time]
        for urgent in urgentOrders:
            urgent.specialFreight = True
            
        
        # iterate over all commission area (origin of origin-destination pair) to book transportation        
        origin = self.setup.originDestination.loc[:,['originX','originY']].apply(lambda row: (row[0], row[1]), axis = 1)
        destination = self.setup.originDestination.loc[:,['destX','destY']].apply(lambda row: (row[0], row[1]), axis = 1)
        transpMode = self.setup.originDestination.loc[:,'transpMode']
        for i in range(len(origin)):
            content = self.grid.get_cell_list_contents(origin[i])
            listOfProducts = [agent for agent in content if isinstance(agent, Product) and agent.intransit is False]
            existingTransport = [agent for agent in content if isinstance(agent, Transportation)]
            if len(existingTransport) > 0:
                minUtilization = min([max(transp.utilizedWeight / transp.weightCapacity * 100, 
                                          transp.utilizedVolume / transp.volumeCapacity * 100) 
                                        for transp in existingTransport])
            else:
                minUtilization = 100
            if len(listOfProducts) == 0 or (minUtilization <  self.rules.minTranspUtilization):
                continue
            else:
                if transpMode[i] == 'land':
                    speed = self.rules.landTranspSpeed
                elif transpMode[i] == 'sea':
                    speed = self.rules.seaTranspSpeed
                elif transpMode[i] == 'air':
                    speed = self.rules.airTranspSpeed
                self.bookTransport(origin[i], destination[i], speed = speed)
    
    def createOrder(self, orderPerLine = True):
        orderLines = self.setup.orderLines[self.setup.orderLines['orderCreateDate'] == self.schedule.time]
        groupCol = orderLines.columns.tolist()
        groupCol.remove('quantity')
        orderLines = orderLines.groupby(groupCol, as_index = False).sum()
        orderLines = orderLines[orderLines['quantity'] > 0]
        
        groupCol.remove('MATERIAL_CODE')
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
            tmp = orderLines.copy().groupby(groupCol, as_index = False)
            for tbl in [tmp.get_group(x) for x in tmp.groups]:
                self.dispatchOrder(tbl, self.rules.dispatchBuffer)

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
        order.name = np.array(orderLines['type'])[0]
        self.num_order = self.num_order + 1
        CRD = min(orderLines['CRD'])
        self.schedule.add(order)
        originRegion = order.originRegion
        destRegion = order.destRegion
        if order.name == 'customerOrder':
            origin = self.setup.whs[self.setup.whs['REGION_EN_NAME'] == originRegion]
        elif order.name == 'materialOrder':
            origin = self.setup.supplier[self.setup.supplier['supplierName'] == originRegion]
        dest = self.setup.whs[self.setup.whs['REGION_EN_NAME'] == destRegion]
        
        if order.name == 'customerOrder':
            # see if order can be directly fulfilled at destination 
            self.allocateProduct(order, (int(dest['x']), int(dest['y'])))
        
        if sum(order.orderLines['quantity']) == 0:
            RPD = CRD
        else:
            # calculate lead time
            # TODO: add distribution to the mean lead time based on actual historical lead time
            transportLT = np.sqrt( np.power(int(origin['x']) - int(dest['x']),2) + np.power(int(origin['y']) - int(dest['y']),2))
            RPD = CRD - int(transportLT) - self.rules.dispatchBuffer
            if RPD < self.schedule.time:
                RPD = self.schedule.time
                order.specialFreight = True
        # place order at origin warehouse
        order.RPD = RPD
        self.grid.place_agent(order, (int(origin['x']), int(origin['y'])))          
        
    def step(self):
        self.createOrder()
        if self.schedule.time == 0:
            self.dispatchFinishedGoods() # if warehouses are initialized with inventory
        """ 
        if self.schedule = SimultaneousActivation(self), 
        then agent.step() and agent.advance() are both executed in self.schedule.step().
        had to change the original mesa code in mesa/time.py class SimultaneousActivation(BaseScheduler)
        to add custom dispatchFinishedGoods() between .step() and .advance() methods, 
        this is to due to execution sequence of Plant.produce() in .step() and ProductionScheduler.readInOrders() in .advance().
        """
         # 
        for agent in self.schedule.agents[:]:
            agent.step()
        
        self.dispatchFinishedGoods()
        
        for agent in self.schedule.agents[:]:
            agent.advance()
        self.schedule.steps += 1
        self.schedule.time += 1
        
        self.datacollector.collect(self)  

#%%

#model = SupplyChainModel()
#
#for i in range(200):
#    model.step()
#agent_counts = np.zeros((model.grid.width, model.grid.height))
#for cell in model.grid.coord_iter():
#    content, x, y = cell
#    agent_counts[x][y] = len(content)
#pd.DataFrame(agent_counts).to_clipboard()
#
#data1 = model.datacollector.get_model_vars_dataframe()

