import random
import re

#ANALSYS: It was oberved that the neighbourhood size again had an interesting effect. Values greater than 0.5
# resulted in high variations and values less than 0.1 were also very inconsistent. Best value for neighbourhood
# size was in between these values. Temperature was started at 1 and then decreased by a factor of 0.8 after every
# 100 iterations. This was the optimum cooling schedule that was observed.

#maxmin takes string as 'maxima' or 'minima'
#neighbourhood size is step-distance
#function takes the input function as a string
#The output is the maxima/minima value
#e.g input : simulatedAnnealing('(1-x)**2 + 100*(y-x**2)**2',-0.5,3.0,-1.5,2.0,0.5,'minima')

def simulatedAnnealing (function,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize,maxmin):

    Tmin = 0.00001
    T = 1
    CoolingConstant = 0.8
    
    #Generate a random point in the given range
    n = num_decimal_places(str(neighbourhoodSize))
    x = round(random.uniform(rangeX0, rangeX1),n)
    y = round(random.uniform(rangeY0, rangeY1),n)

    #Calculate cost of the initial solution
    value0 = eval(function)

    #Calculates co-ordinates and cost of neighbour
    lst = generateRandomNeighbour (x,y,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize)
    value = calculateCost(function,lst[0],lst[1])

    #Runs till Temperature crosses a certain threshold
    while (T >= Tmin):

        for i in range(0,100):
            if (compareCost(value0,value,T,maxmin)):
                
                value0 = value
                x = lst[0]
                y = lst[1]
                lst = generateRandomNeighbour (x,y,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize)
                value = calculateCost(function,lst[0],lst[1])
                
            else:
                lst = generateRandomNeighbour (x,y,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize)
                value = calculateCost(function,lst[0],lst[1])
               
        T=T*CoolingConstant


    return value0

#Checks whether to progress    
def compareCost(value0, value, temperature,maxmin):

    if maxmin == 'maxima':
        if value > value0:
            return True

        elif value <= value0:
            return acceptanceProbabilty(value0, value, temperature,maxmin)
        
    elif maxmin == 'minima':
        if value < value0:
            return True

        elif value >= value0:
            return acceptanceProbabilty(value0, value, temperature,maxmin)
       
#Calculates acceptance probability based on which outputs true or false        
def acceptanceProbabilty(value0, value, temperature,maxmin):

    if maxmin == 'maxima':
        ap = (2.71828**((value-value0)/temperature))
        r = random.uniform(0, 1)

    elif maxmin == 'minima':
        ap = (2.71828**((value0-value)/temperature))
        r = random.uniform(0, 1)
        
    if r >= ap:
        return True
    else:

        return False

#Calculates the cost of function (evaluates function at a given value of x and y)    
def calculateCost(function,x0,y0):
    y=y0
    x=x0
    return eval(function)



#Generates a random neighbour from a given point
def generateRandomNeighbour (x,y,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize):

    x0 = round((x - neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
    y0 = round((y - neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
    
    i = random.randint(0,2)
    j = random.randint(0,2)
    
    x = round((x0 + i*neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
    y = round((y0 + j*neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))

    if (x < rangeX0 or x > rangeX1 or y < rangeY0 or y > rangeY1 or (i==1 and j==1)):
        
        x0 = round((x0 + neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
        y0 = round((y0 + neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
        return generateRandomNeighbour (x0,y0,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize)

    else:

        return x,y






#helper function to calculate number of decimal places
def num_decimal_places(value):
    m = re.match(r"^[0-9]*\.([1-9]([0-9]*[1-9])?)0*$", value)
    return len(m.group(1)) if m is not None else 0
