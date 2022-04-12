import cv2
#from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np
import math
#constants
DEFAULT_SIZE = (256,256)
ORGIN = (DEFAULT_SIZE[0]//2,DEFAULT_SIZE[1]//2) #pixel space location of orgin
FUNCTION_SCALE = 0.1 #0.1 in fxn scale per pixel

def cart_test(x,y): #test fxn for cartesian
    return x**2 - y

def polar_test(theta): #test fxn for polar
    return math.sin(theta)

#plot a point on the array at x,y where x,y is coords from 0-size of image
#but, they are oriented to make a graph look correct
def plot_point(map,x,y,color=(255,255,255)):
    map[DEFAULT_SIZE[1] - 1 - y,x] = color

#plot polar function-iterate over theta
def plot_polar(map,f,color=(255,255,255)):
    theta = 0
    T_STEP = 0.001
    NUM_ROT = 5
    while theta < NUM_ROT * 2 * math.pi:
        n = f(theta)
        #cos(theta) outputs in function space, so we divide by
        #fxn scal/pixel to get pixel space, then add the
        #coordinates (in pixel space) of the origin
        x = int(math.cos(theta) * n / FUNCTION_SCALE + ORGIN[0])
        y = int(math.sin(theta) * n / FUNCTION_SCALE + ORGIN[1])
        #get rect coords of f(theta)

        #if its in the img
        if x >= 0 and x < DEFAULT_SIZE[0]:
            if y >= 0 and y < DEFAULT_SIZE[1]:
                plot_point(map,x,y,color)
                #plot
                
        theta += T_STEP
    
def plot_cartesian(map,f,color = (255,255,255)):
    corners = np.zeros((DEFAULT_SIZE[0],DEFAULT_SIZE[1]), np.uint8)
    
    for x in range(DEFAULT_SIZE[0]):
        for y in range(DEFAULT_SIZE[1]):
            #convert x and y (pixel space) to fxn space
            if f((x - ORGIN[0])*FUNCTION_SCALE,(y - ORGIN[1])*FUNCTION_SCALE) > 0:
                corners[x,y] = 1
                #OK so this here is a bit interesting.
                #See function cart_test(x,y) to understand a bit better
                #for x**2 = y, to graph,
                #just color all points where x**2 - y == 0 right? (for every x and y in fxn space)
                # wrong. This only colors perfect squares
                #so, what you must do is find boundary pixels between x**2 > y and y > x**2,
                #since between those 2 is a 0 and thus a sxn to the
                #exn is within this square, thus color
                #thats whats held in this array, and the next loop colors all pixels by this method
    for x in range(DEFAULT_SIZE[0]-1):
        for y in range(DEFAULT_SIZE[1]-1):
            a = corners[x,y]
            b = corners[x+1,y]
            c = corners[x+1,y+1]
            d = corners[x,y+1]
            if (a!=b or a!=c or a!=d):
                plot_point(map,x,y,color)

def engorge(map,rad = 2):
    newMap = np.zeros(map.shape,np.uint8)
    for x in range(DEFAULT_SIZE[0]):
        for y in range(DEFAULT_SIZE[1]):
            if map[x,y,0] != 0:
                newMap[x,y] = (255,255,255)
            else:
                cont = True
                for xoff in range(-rad,rad + 1):
                    for yoff in range(-rad,rad + 1):
                        if (x+xoff >= 0 and x+xoff < DEFAULT_SIZE[0] and y+yoff >=0 and y + yoff < DEFAULT_SIZE[1]):
                            if (map[x+xoff,y+yoff,0] != 0):
                                newRad = xoff**2 + yoff**2
                                if newRad < rad**2:
                                    newMap[x,y] = (255,255,255)
                                    cont = False
                        if not cont:
                            break
                    if not cont:
                        break
    
    #for exvery pixel, if it has a white pixel within rad ditance, color it white
    #this engorges the normally very thin graph lines to
    #thick lines-a track. 
    return newMap


def makeImageFromFunc(name,cartFunc,polarFunc,orgin,pixelsToMeters=1,filepath=""):
    blank = np.zeros((DEFAULT_SIZE[0],DEFAULT_SIZE[1],3), np.uint8)
    #blank.fill(255)
    plot_cartesian(blank,lambda x,y:y==0)
    for c in cartFunc:
        plot_cartesian(blank,c)
    for p in polarFunc:
        plot_polar(blank,p)
    blank = engorge(blank)
    #cv2.imshow("Eng",blank)
    #take engorged track image, apply canny edge to get track edges
    edges = cv2.Canny(image=blank, threshold1=100, threshold2=200)
    #cv2.imshow('Canny Edge Detection', edges)
    edges = 255 - edges
    #canny returns with thin white lines on black bckgrnd
    #so invert colors to fulfill formatting for f1tenths
    cv2.imshow('Final map', edges)
    if (filepath != "" and filepath[-1] != "/" and filepath[-1] != "\\"): filepath = filepath + "/"

    cv2.imwrite(filepath + name+".png", blank)
    fobj = open(filepath + name+".yaml","w+")
    fobj.write("image: "+name+".png\n")
    fobj.write("resolution: "+str(pixelsToMeters)+"\n")
    orgin.append(0.000)
    fobj.write("orgin: " + str(orgin)+"\n")
    fobj.write("""
negate: 0
occupied_thresh: 0.45
free_thresh: 0.196
""")
    fobj.close()
makeImageFromFunc("LOOP",[],[lambda theta:10],[0,0],1)
