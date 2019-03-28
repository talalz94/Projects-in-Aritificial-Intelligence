import random
import re

#ANALYSIS: It was noted that the smaller the neighbourhoodSize, the better/ more consistent the result got. However
# below neighbourSize of 0.1, the result got very inconsisten and the variation between results increased for different
#iterations. Best range for the neighbourhoodSize for given function was 0.1 < and > 0.5.


#maxmin takes string as 'maxima' or 'minima'
#neighbourhood size is step-distance
#function takes the input function as a string
#The output is in the form (maxima/minima value, x, y)
#e.g input : hillClimbing('(1-x)**2 + 100*(y-x**2)**2',-0.5,3.0,-1.5,2.0,0.5,'minima')


def hillClimbing (function,rangeX0,rangeX1,rangeY0,rangeY1,neighbourhoodSize,maxmin):
    n = num_decimal_places(str(neighbourhoodSize))
    x = round(random.uniform(rangeX0, rangeX1),n)
    y = round(random.uniform(rangeY0, rangeY1),n)

    value0 = eval(function)
    #print(value0)
    calculateNeighbour(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1,value0,neighbourhoodSize,maxmin)

  

    
#Calculates value of a function then compares it with its neighbours, if there is a better value, it moves to that and recursively checks its neighbour until
#no better value is present
def calculateNeighbour(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1,value0,neighbourhoodSize,maxmin):
    x = round((x - neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
    y = round((y - neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
    maxX = x
    maxY = y
    

    value = value0

    #Checks for every valid neighbour around a particular point
    for i in range (0,3):
        y = round((y + i*neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))
        for j in range (0,3):

            x = round((x + j*neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))

            
            if not (x < rangeX0 or x > rangeX1 or y < rangeY0 or y > rangeY1):

                if (maxmin == 'maxima'):
                    if ((calculateValue(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1) > value)):

                        value = (calculateValue(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1))
                        maxX = x
                        maxY = y
                        
                if (maxmin == 'minima'):
                    if ((calculateValue(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1) < value)):

                        value = (calculateValue(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1))
                        maxX = x
                        maxY = y   

                
            x = round((x - j*neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))   
                
        y = round((y - i*neighbourhoodSize), num_decimal_places(str(neighbourhoodSize)))

        
    if (maxmin == 'maxima'):
        if (value > value0):
            calculateNeighbour(function,maxX,maxY,rangeX0,rangeX1,rangeY0,rangeY1,value,neighbourhoodSize,maxmin)
        else:
            print(value, maxX, maxY)
            
    
    if (maxmin == 'minima'):
        if (value < value0):
            calculateNeighbour(function,maxX,maxY,rangeX0,rangeX1,rangeY0,rangeY1,value,neighbourhoodSize,maxmin)
        else:
            print(value, maxX, maxY)


#helper function to calculate number of decimal places           
def num_decimal_places(value):
    m = re.match(r"^[0-9]*\.([1-9]([0-9]*[1-9])?)0*$", value)
    return len(m.group(1)) if m is not None else 0

#Calculates the cost of function (evaluates function at a given value of x and y)    
def calculateValue(function,x,y,rangeX0,rangeX1,rangeY0,rangeY1):


        return eval(function)
        





