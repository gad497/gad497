import cv2
import os
import numpy as np
import time
import math
#%%
# for selection of corner points
def pt_select(event,x,y,flags,param):
    global ref_pt
    if event==cv2.EVENT_LBUTTONDOWN:
        ref_pt.append((x,y))
        print(ref_pt)
#%%
# to find slope of the line
def slope(x1,y1,x2,y2):
    m=((y2-y1)/(x2-x1))
    return m
#%%
# to find intercept of two lines
def find_intercept(x1,y1,x2,y2,m1,m2):
    c1=y1-m1*x1
    c2=y2-m2*x2
    x=((c2-c1)/(m1-m2))
    y=m1*x+c1
    return([x,y])
#%%
# to find euclidian distance
def edist(x1,y1,x2,y2):
    d=math.sqrt((y2-y1)**2+(x2-x1)**2)
    return d
#%%
vid=cv2.VideoCapture('C:\\Users\\Glen Avin Dsilva\\Desktop\\MIT\\2nd sem\\MVIP\\mini project\\Beachfront.mkv')
#vid=cv2.VideoCapture(0)
#create a folder to store each frame if needed
try:
    if not os.path.exists('img'):
        os.makedirs('img')
except:
    print('unable to create folder')
os.chdir('C:\\Users\\Glen Avin Dsilva\\Desktop\\MIT\\2nd sem\\MVIP\\mini project\\img')
#declare variables needed for the task
fgbg = cv2.createBackgroundSubtractorMOG2()  #background subtractor
cf=0 # current frame number
uid=0 # unique id
ids={} # dictionary with all ids
fcnt=5 # frame count for deleting ids when out of frame
ref_pt=[] # corner points of ROI in list
vel={} # points for calculation of velocity
speed={} # final speed of vehicles with id as key
width=vid.get(cv2.CAP_PROP_FRAME_WIDTH) # width of the frame
height=vid.get(cv2.CAP_PROP_FRAME_HEIGHT) #height of the frame
fps=vid.get(cv2.CAP_PROP_FPS) # frames per second for calculating time
# kernels for morphological operations 
kernel = np.ones((7,7),np.uint8) 
kernel2=np.ones((3,3),np.uint8)
kernel3=np.ones((11,11),np.uint8)
kernel4=np.array([[0,1,0],[1,1,1],[0,1,0]])
# corner points selection (top left, bottom left, bottom right, top right)
ret,image=vid.read()
cv2.namedWindow('win')
cv2.setMouseCallback('win',pt_select)
while True:
    cv2.imshow('win',image)
    if len(ref_pt)==4:
        cv2.destroyWindow('win')
        break
    if cv2.waitKey(10)&0xFF==27:
        break
# input the length and breadth of the rectangular area
hdist=float(input('enter horizontal distance in meters:'))
vdist=float(input('enter vertical distance in meters:'))
# finding the vanishing points if they exist
m1=slope(ref_pt[0][0],ref_pt[0][1],ref_pt[1][0],ref_pt[1][1])
m2=slope(ref_pt[2][0],ref_pt[2][1],ref_pt[3][0],ref_pt[3][1])
m3=slope(ref_pt[1][0],ref_pt[1][1],ref_pt[2][0],ref_pt[2][1])
m4=slope(ref_pt[0][0],ref_pt[0][1],ref_pt[3][0],ref_pt[3][1])
vv=True
hv=True
if m1==m2:
    vv=False
if m3==m4:
    hv=False
if vv==True:
    vvpt=find_intercept(ref_pt[0][0],ref_pt[0][1],ref_pt[3][0],ref_pt[3][1],m1,m2)
if hv==True:
    hvpt=find_intercept(ref_pt[0][0],ref_pt[0][1],ref_pt[1][0],ref_pt[1][1],m4,m3)
    
while True:
    ret,frame=vid.read()
    if ret:
        # storing of frames
        name = 'C:/Users/Glen Avin Dsilva/Desktop/MIT/2nd sem/MVIP/mini project/img/frame' + str(cf) + '.jpg'
        #cv2.imwrite(name, frame) 
        gimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # convert to gray scale
        fgimg=fgbg.apply(gimg) # applying background substraction
        retf,fgimg=cv2.threshold(fgimg,150,255, cv2.THRESH_BINARY)
        if cf < 2:
            if cf==0:
                gimg2=gimg
            if cf==1:
                gimg1=gimg
            cf+=1
            continue
        #subtraction of the frames
        sub1=np.absolute(np.subtract(gimg1,gimg))
        sub1 = cv2.morphologyEx(sub1, cv2.MORPH_CLOSE, kernel)
        sub2=np.absolute(np.subtract(gimg2,gimg))
        sub2 = cv2.morphologyEx(sub2, cv2.MORPH_CLOSE, kernel)
        # convert subtracted frames to binary
        ret1,thresh1=cv2.threshold(sub1,200,255, cv2.THRESH_BINARY)
        ret2,thresh2=cv2.threshold(sub2,200,255, cv2.THRESH_BINARY)
        # performing logical and
        logic=np.logical_and(thresh1,thresh2)
        thresh3=np.multiply(logic,255)
        thresh3=thresh3.astype(np.uint8)
        logic1=np.logical_and(thresh3,fgimg)
        thresh4=np.multiply(logic1,255)
        thresh4=thresh4.astype(np.uint8)
        # applying morphological operations
        thresh4=cv2.morphologyEx(thresh4, cv2.MORPH_CLOSE, kernel3)
        thresh5 = cv2.erode(thresh4,kernel2,iterations = 1)
        thresh5 = cv2.dilate(thresh5,kernel2,iterations = 14)
        thresh5 = cv2.erode(thresh5,kernel,iterations = 2)
        cv2.imshow('thresh5',thresh5)
        # finding contours
        contour,hir=cv2.findContours(thresh5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tracker=[]
        # labelling and tracking
        for cnt in contour:
            x,y,w,h=cv2.boundingRect(cnt)
            status=False
            # assigning the id to next bounding rectangle if id exists for that vehicle
            for val in ids:
                xp,yp,fcntp=ids[val]
                if ((x>=((xp-(width/8))) and (x<=((xp+(width/8))))) and (y>=((yp-(height/8))) and (y<=((yp+(height/8)))))):
                    ids[val]=[x,y,fcnt]
                    tracker.append(val)
                    dispid=val
                    status=True
                    break
            # assigning new id for vehicles detected
            if status==False:
                uid+=1
                ids[uid]=[x,y,fcnt]
                tracker.append(uid)
                dispid=uid
            # remove ids that left the frame
            if (x<100 and x>(width-100)) and (y<100 and y>(height-100)):
                for i in ids:
                    if ids[i][0]==x and ids[i][1]==y:
                        del ids[i]
            #remove ids that disappeared in between
            if len(contour)<len(ids):
                z=[]
                for i in ids:
                    if i not in tracker:
                        z.append(i)
                for i in z:
                    ids[i][2]-=1
                    if i in ids and i in vel and ids[i][2]==0:
                        del ids[i],vel[i]
            # display bounding rectangles with id and speed if calculated
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
            if dispid in speed:
                cv2.putText(frame,'ID:'+str(dispid)+'  speed:'+str(speed[dispid]),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
            else:
                cv2.putText(frame,'ID:'+str(dispid),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
        # Speed calculation for every 5 frames
        if cf%5==0:
            for i in ids:
                # store the position 
                if i in vel.keys():
                    vel[i].append([ids[i][0],ids[i][1]])
                else:
                    vel[i]=[[ids[i][0],ids[i][1]]]
                # calculating true distance travelled in the interval
                if i in vel and len(vel[i])>1:
                    l=len(vel[i])
                    if hv==True:
                        mh1=slope(hvpt[0],hvpt[1],vel[i][l-2][0],vel[i][l-2][1])
                        mh2=slope(hvpt[0],hvpt[1],vel[i][l-1][0],vel[i][l-1][1])
                    else:
                        mh1=m3
                        mh2=m3
                    if vv==True:
                        mv1=slope(vvpt[0],vvpt[1],vel[i][l-2][0],vel[i][l-2][1])
                        mv2=slope(vvpt[0],vvpt[1],vel[i][l-1][0],vel[i][l-1][1])
                    else:
                        mv1=m1
                        mv2=m1
                    in1=find_intercept(vel[i][l-2][0],vel[i][l-2][1],ref_pt[0][0],ref_pt[0][1],mh1,m1)
                    in2=find_intercept(vel[i][l-2][0],vel[i][l-2][1],ref_pt[3][0],ref_pt[3][1],mh1,m2)
                    in3=find_intercept(vel[i][l-2][0],vel[i][l-2][1],vel[i][l-1][0],vel[i][l-1][1],mh1,mv2)
                    d1=edist(in1[0],in1[1],in2[0],in2[1])
                    d2=edist(vel[i][l-2][0],vel[i][l-2][1],in3[0],in3[1])
                    xspeed=((d2*hdist*fps)/(d1*5))
                    
                    in4=find_intercept(vel[i][l-2][0],vel[i][l-2][1],ref_pt[0][0],ref_pt[0][1],mv1,m4)
                    in5=find_intercept(vel[i][l-2][0],vel[i][l-2][1],ref_pt[1][0],ref_pt[1][1],mv1,m3)
                    in6=find_intercept(vel[i][l-2][0],vel[i][l-2][1],vel[i][l-1][0],vel[i][l-1][1],mv1,mh2)
                    d3=edist(in4[0],in4[1],in5[0],in5[1])
                    d4=edist(vel[i][l-2][0],vel[i][l-2][1],in6[0],in6[1])
                    yspeed=((d4*vdist*fps)/(d3*5))
                    # final speed of vehicle
                    speed[i]=math.trunc(math.sqrt(xspeed**2+yspeed**2)*3.6)
        # display region of interest            
        cv2.line(frame,ref_pt[0],ref_pt[1],(0,255,0))
        cv2.line(frame,ref_pt[1],ref_pt[2],(0,255,0))
        cv2.line(frame,ref_pt[2],ref_pt[3],(0,255,0))
        cv2.line(frame,ref_pt[3],ref_pt[0],(0,255,0))
        cv2.imshow('frame',frame)
        # saving current frame as previous frame for next itteration
        gimg2=gimg1
        gimg1=gimg
        time.sleep(1/fps) # syncronize the speed of operation with frame speed
        cf += 1
    else: 
        break
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
vid.release() 
cv2.destroyAllWindows()
#%%
