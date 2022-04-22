import cv2

#pip install opencv-python

#from PIL import Image
#from matplotlib import pyplot as plt
import numpy as np
import math
import random
import copy

#constants
DEFAULT_SIZE = (700,700)
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

def plot_pix_line(map,pt1,pt2):
    corners = np.zeros((DEFAULT_SIZE[0],DEFAULT_SIZE[1]), np.uint8)
    m = (pt1[1]-pt2[1])
    den = pt1[0]-pt2[0]
    if den == 0:
        m = 1000000
    else:
        m /= den
    b = pt1[1] - (m * pt1[0])
    xR = -1 if pt1[0] > pt2[0] else 1
    yR = -1 if pt1[1] > pt2[1] else 1
    for x in range(pt1[0]-xR,pt2[0] + xR,xR):
        for y in range(pt1[1]-yR,pt2[1] + yR,yR):
            if (m*x+b) - y < 0:
                corners[x,y] = 1
    #cv2.imshow("corn", corners)
    for x in range(pt1[0],pt2[0],xR):
        for y in range(pt1[1],pt2[1],yR):
            a = corners[x,y]
            b = corners[x+1,y]
            c = corners[x+1,y+1]
            d = corners[x,y+1]
            if (a!=b or a!=c or a!=d):
                plot_point(map,x,y,(255,255,255))

def plot_circle_segment(map,pt1,pt2,ctr):
    corners = np.zeros((DEFAULT_SIZE[0],DEFAULT_SIZE[1]), np.uint8)
    radiusSQ = (ctr[0]-pt1[0])**2 + (ctr[1]-pt1[1])**2
    xR = -1 if pt1[0] > pt2[0] else 1
    yR = -1 if pt1[1] > pt2[1] else 1
    for x in range(pt1[0] - xR,pt2[0] + xR,xR):
        for y in range(pt1[1]-yR,pt2[1] + yR,yR):
            if (x-ctr[0])**2 + (y-ctr[1])**2 < radiusSQ:
                corners[x,y] = 1
    for x in range(pt1[0],pt2[0], xR):
        for y in range(pt1[1],pt2[1],yR):
            a = corners[x,y]
            b = corners[x+1,y]
            c = corners[x+1,y+1]
            d = corners[x,y+1]
            if (a!=b or a!=c or a!=d):
                plot_point(map,x,y,(255,255,255))
         
def engorge(map,itr = 20): #thickens lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dilate = cv2.dilate(map, kernel, iterations=itr)
    return dilate



#orgin is car coords at the beginning, in pixel space
#you may need to play with orgin values to find one that is actually within the track
def makeImageFromFunc(name,cartFunc,polarFunc,orgin,pixelsToMeters=0.07712,filepath="",show = True):
    blank = np.zeros((DEFAULT_SIZE[0],DEFAULT_SIZE[1],3), np.uint8)
    #blank.fill(255)
    #plot_cartesian(blank,lambda x,y:y==0)

    if (filepath != "" and filepath[-1] != "/" and filepath[-1] != "\\"): filepath = filepath + "/"

    
    print("Graphing functions...")
    for c in cartFunc:
        plot_cartesian(blank,c)
    for p in polarFunc:
        plot_polar(blank,p)
    print("Thickening lines...")
    blank = engorge(blank)
    
    #cv2.imshow("Eng",blank)
    #take engorged track image, apply canny edge to get track edges
    print("Getting edges...")
    edges = cv2.Canny(image=blank, threshold1=100, threshold2=200)
    #cv2.imshow('Canny Edge Detection', edges)

    #add edges to image, to keep car from escaping
    for row in range(DEFAULT_SIZE[0]):
        edges[row,0] = 255
        edges[row,DEFAULT_SIZE[1] - 1] = 255
    for col in range(DEFAULT_SIZE[1]):
        edges[DEFAULT_SIZE[0] - 1,col] = 255
        edges[0,col] = 255


    #canny returns with thin white lines on black bckgrnd
    #so invert colors to fulfill formatting for f1tenths
    edges = engorge(edges,1)
        
    edges = 255 - edges

    disp = edges.copy()
    disp[orgin[1],orgin[0]] = 0

    if (show):
        cv2.imshow('Final map', disp)

 #for some reason negative meters moves the car right

    orgin[0] = orgin[0] * -pixelsToMeters
    orgin[1] = orgin[1] * -pixelsToMeters
    

    cv2.imwrite(filepath + name+".png", edges)
    fobj = open(filepath + name+".yaml","w+")
    fobj.write("image: "+name+".png\n")
    fobj.write("resolution: "+str(pixelsToMeters)+"\n")
    orgin.append(0.000)
    fobj.write("origin: " + str(orgin))
    fobj.write("""
negate: 0
occupied_thresh: 0.45
free_thresh: 0.196
starting_angle: 0
default_resolution: 0.07712
""")
    fobj.close()

def skele(img):
    
    # Step 1: Create an empty skeleton
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    # Repeat steps 2-4
    while True:
        #Step 2: Open the image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        #Step 3: Substract open from the original image
        temp = cv2.subtract(img, open)
        #Step 4: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
        # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img)==0:
            break
    return skel

def flood(map,pt):
    h, w = map.shape[:2]

    mask = np.zeros((h+2, w+2), np.uint8)
    #Floodfill from point (0, 0)
    cv2.floodFill(map, mask, pt, 0); #fill image
    cv2.floodFill(map, mask, (0,0), 255); #remove border
    
#saves image to be cleaned
def make_graph(filename):
    map = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)
    map = cv2.addWeighted( map, 100, map, 0, -500)
    #cv2.imshow("m",map)
    #cv2.waitKey(0)
    flood(map,(650,500))
    img = skele(255 - map)
    #img = # call addWeighted function. use beta = 0 to effectively only operate one one image
    img = cv2.addWeighted( img, 100, img, 0, -500)
    img = skele(img)
    cv2.imshow("skel", img)
    print("please clean up "+filename + " and remove and small deviations from the idea path.")
    cv2.imwrite("GRAPH_"+filename,img)

def get_point(points,t):
    p0, p1, p2, p3 = (0,0,0,0);

   
    p1 = int(t);
    p2 = (p1 + 1) % len(points);
    p3 = (p2 + 1) % len(points);
    p0 = (p1 - 1) % len(points);
    
    t = t - int(t)

    tt = t * t;
    ttt = tt * t;

    q1 = (-ttt) + (2.0*tt) - t;
    q2 = (3.0*ttt) - (5.0*tt) + 2.0;
    q3 = (-3.0*ttt) + (4.0*tt) + t;
    q4 = ttt - tt  

    tx = 0.5 * (points[p0][0] * q1 + points[p1][0] * q2 + points[p2][0] * q3 + points[p3][0] * q4);
    ty = 0.5 * (points[p0][1] * q1 + points[p1][1] * q2 + points[p2][1] * q3 + points[p3][1] * q4);

    return ( int(tx), int(ty) );

def draw_splines(map,points):
    #points = [(650,150), (550,250), (550, 500), (650,650),(680,600)]
    col = (255)
    if (len(map.shape) == 3):
        col = (0,0,255)

    #disp points
    for i in points:
        cv2.circle(map,(int(i[0]),int(i[1])),2,col,3)
    coords = get_point(points,0)
    t = 0.001   
    while (t < len(points)):
        last = coords
        coords = get_point(points,t)
        cv2.line(map,last,coords,col,1)
        t += 0.001
    #cv2.imshow('spline', map)

def place_between(lis, index,dx = 0,dy = 0):
    return lis[:index + 1]+[(dx + (lis[index][0] + lis[index+1][0])//2 , dy + (lis[index][1] + lis[index+1][1])//2)] + lis[index + 1:]

def translate(newL,index,dx=0,dy =0):
    newL[index] = (newL[index][0] + dx, newL[index][1] +dy)
    

def extract_line(filename):
    map = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2GRAY)
    map = cv2.addWeighted( map, 100, map, 0, -500)
    colMap = cv2.imread(filename[len("GRAPH_"):])
    
    #cv2.imshow("map", map)

    
    #place red pixels at initial splice points    
    spline_pts = np.argwhere(  (colMap == (0,0,255)).all(axis = 2))
    nl = []
    for i in spline_pts:
        nl.append((i[1],i[0]))
    #draw_splines(colMap,nl)
    #cv2.imwrite("ordering.png",colMap)
    # for SILVERSTONE_OBS ONLY
    reOrd = [0,2,8,9,12,14,17,18,23,22,20,21,19,16,15,11,13,10,4,3,7,6,5,1]
    newL = [nl[i] for i in reOrd]
    newL[8] = (newL[8][0]+30, newL[8][1] - 50)
    #newL = place_between(newL,8,dy = 25)
    #newL = place_between(newL,8,dy = 25)
    translate(newL,0,-25,-10)
    translate(newL,-7,-30,-40)
    translate(newL,-12,20)
    translate(newL,-14,20,30)
    #draw_splines(colMap,newL)
    #print(nl)
    return newL
    #cv2.imshow("main",colMap)
    #cv2.imwrite("output.png",colMap)
    
def spline_length(points):
    coords = get_point(points,0)
    t = 0.01
    dist = 0
    while (t < len(points)):
        last = coords
        coords = get_point(points,t)
        dist += np.sqrt( (coords[0] - last[0])**2 + (coords[1] - last[1])**2)
        t += 0.01
    return dist

def genetic(track,points,f,alpha):
    newL = np.copy(points)
    for x in range(track.shape[0]):
        for y in range(track.shape[1]):
            if track[y,x][2] != track[y,x][1]:
                track[y,x] = (255,255,255)
    # remove red waypoints
    
    cv2.imshow("main",track)
    #cv2.waitKey(0)
    track = cv2.cvtColor(track,cv2.COLOR_BGR2GRAY)
    track = 255 - track
    map = track.copy()
    cv2.imshow("main",map)
    smallest = 1028289
    for _ in range(100):
        while (cv2.countNonZero(map & track) != 0):
            map *= 0
            newL = copy.deepcopy(points)
            translate(newL,random.randint(0,len(newL)-1),alpha * (random.randint(-100,100))/100,alpha * (random.randint(-100,100))/100)
            draw_splines(map,newL)
            map = engorge(map,1)
        if spline_length(newL) < smallest:
            points = np.copy(newL)
    draw_splines(map,points)
    cv2.imshow("main",map | track)
    return points,(map | track)
    #cv2.imshow('map', map)
    #cv2.imwrite("mappp.png",map)
    #cv2.imshow("main",map & track)
    #cv2.imwrite("save.png",map&track)
    #
    
lClick = False
def click(event, x, y, flags, param):
    global lClick
    if event == cv2.EVENT_LBUTTONDOWN and not lClick:
        lClick = True
    elif event == cv2.EVENT_LBUTTONUP and lClick:
        lClick = False
	# record the ending (x, y) coord

cv2.namedWindow("main")
cv2.setMouseCallback("main", click)

spline_pts = extract_line("GRAPH_SILVERSTONE_OBS.png")
genetic(cv2.imread("SILVERSTONE_OBS.png"),spline_pts,1,0.2)
#[(1295, 105), (807, 167), (1414, 190), (803, 353), (904, 370), (672, 383), (664, 480), (725, 511), (1475, 563), (1516, 722), (1374, 798), (1262, 830), (1484, 868), (1332, 946), (1529, 946), (1022, 971), (790, 981), (1545, 1045), (1430, 1156), (475, 1411), (657, 1529), (576, 1579), (890, 1839), (1049, 1852)]


#makeImageFromFunc("STD",[],[],[645,350],0.07712,show = True)

#blank = np.zeros((DEFAULT_SIZE[0],DEFAULT_SIZE[1],3), np.uint8)
#plot_circle_segment(blank,(0,255),(255,0),(255,255))
#cv2.imshow("blank",blank)
