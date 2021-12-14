import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from pymavlink import mavutil
from imutils.video import WebcamVideoStream
import collections
import csv
from datetime import datetime
from scipy.spatial.transform import Rotation
import os
import time
import threading
import warnings

fx=0.5#Image x scaling
fy=0.5 #Image y scaling
c=50 #Adaptive Thresholding Constant 
Area1=1000*fx #Area of outer contour
Area2=1000*fx#1000 #Area of inner contour
ccl=0.6 #circularity Index
K=np.array([[1.65e+03, 0.00000000e+00, 3.19733649e+02],
        [0.00000000e+00, 1.65e+03, 2.43725445e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])#camera calibration matrix
offset=np.array([-0.05,0.25,0],dtype=float)#offset of uav center from camera center (defined in camera frame)
con=180/math.pi
display=True
logging=True
plotting=True


def findH(gray_thresh,cnts,h,marker_found):
    mask = np.zeros_like(gray) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, cnts, h, (255,255,255), -1) # Draw filled contour in mask
    out=cv2.bitwise_and(255-gray_thresh,mask)#and operation to extract H
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, 
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))#remove outer noise
    _,cnts1, hier1= cv2.findContours(out, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    for cont in cnts1:
        #finding approximate polygon for the contour
        approx=cv2.approxPolyDP(cont,0.01*cv2.arcLength(cont,True),True)
        #extracting x and y values
        x, y = approx[:,-1, 0], approx[:, -1, 1]
        # get the kernel that you will sum around your corner points
        kernel = np.float64(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))###13,13
        kernel /= np.sum(kernel)
        # convolve the image with the kernel, and pull out the sums at the corner points
        conv = cv2.filter2D(out/255, cv2.CV_64F, kernel)
        neighborhood_sums = conv[y, x]
        # concave indices have more white than black around them, so convolution will be larger
        concave_indices = neighborhood_sums > 0.7
        convex_indices = neighborhood_sums <0.55
        print(np.count_nonzero(concave_indices==True),np.count_nonzero(convex_indices==True))
        if np.count_nonzero(concave_indices==True)==4 and np.count_nonzero(convex_indices==True)==8:
            #print('\nmarker found')
            marker_found=True
            break
    return marker_found
    
def Rotation_matrix(psi,theta,phi):
    #create rotation matrix
    ctheta=math.cos(theta)
    stheta=math.sin(theta)
    cphi=math.cos(phi)
    sphi=math.sin(phi)
    cpsi=math.cos(psi)
    spsi=math.sin(psi)
    Rx=np.array([[1,0,0],
                [0,cpsi,-spsi],
                [0,spsi,cpsi]])
    Ry=np.array([[ctheta,0,stheta],
                [0,1,0],
                [-stheta,0,ctheta]])
    Rz=np.array([[cphi,-sphi,0],
                [sphi,cphi,0],
                [0,0,1]])
    Rd=np.matmul(Rx,np.matmul(Ry,Rz))
    return Rd

def Localize(u,v,Rd,h,K,offset):
    X=np.array([0,0,h])
    U=np.array([u,v,1])
    M=np.matmul(np.linalg.inv(K),U)
    Ad=np.array([[M[0],-Rd[0,0],-Rd[0,1]],
                [M[1],-Rd[1,0],-Rd[1,1]],
                [M[2],-Rd[2,0],-Rd[2,1]]])
    Yd=np.matmul(np.linalg.inv(Ad),np.matmul(Rd,X)+offset)
    return Yd
    
class Drawing:
    def __init__(self):
        self.img=None
    def draw(self,img,imgpts):
        corner = tuple(imgpts[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (255,0,0), 2)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,255,0), 2)
        img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (0,0,255), 2)
        return img        
    def draw_frame(self,img,Rd,h,K,offset,fx):
        dist=np.zeros(5)
        #dist=np.array([ 2.41799209e-02,  1.69643240e-01,  2.55195149e-03, -4.20323123e-04,  3.03453213e+01])
        r=Rotation.from_matrix(Rd)
        rvecs=r.as_rotvec()
        tvecs=np.array([[0,0,h]],dtype=float)
        #print('dist:',dist,'\nrvecs:',rvecs,'\ntvecs:',tvecs)

        axis=np.array([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]]).reshape(-1,3)
        imgpts,_=cv2.projectPoints(axis,rvecs,tvecs,K,dist)
        imgpts=(imgpts*fx).astype(int)
        #print(imgpts)
        img=self.draw(img,imgpts)

        tvecs2=np.array([[Rd[0,2]*h+offset[0],Rd[1,2]*h+offset[1],Rd[2,2]*h+offset[2]]],dtype=float)
        #tvecs2=np.matmul(Rd,tvecs2).T
        imgpts2,_=cv2.projectPoints(axis,rvecs,tvecs2,K,dist)
        imgpts2=(imgpts2*fx).astype(int)
        img=self.draw(img,imgpts2)

        img=cv2.putText(img,'img_center',tuple(imgpts[0].ravel()),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1,cv2.LINE_AA)
        img=cv2.putText(img,'uav_center',tuple(imgpts2[0].ravel()),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1,cv2.LINE_AA)
        #cv2.imshow('check',img)
        return img
    def draw_frame_2(self,img,Rd,Yd,h,K,offset,fx):
        dist=np.zeros(5)
        #dist=np.array([ 2.41799209e-02,  1.69643240e-01,  2.55195149e-03, -4.20323123e-04,  3.03453213e+01])
        r=Rotation.from_matrix(Rd)
        rvecs=r.as_rotvec()
        Yd=np.array([[Yd[1],Yd[2],h]],dtype=float).T
        tvecs3=np.matmul(Rd,Yd)+np.array([offset]).T
        #tvecs3=np.matmul(Rd,tvecs3).T
        axis=np.array([[0,0,0],[0.1,0,0],[0,0.1,0],[0,0,0.1]]).reshape(-1,3)
        imgpts3,_=cv2.projectPoints(axis,rvecs,tvecs3,K,dist)
        imgpts3=(imgpts3*fx).astype(int)
        img=self.draw(img,imgpts3)
        img=cv2.putText(img,'target',tuple(imgpts3[0].ravel()),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,255),1,cv2.LINE_AA)
        return img
    def drawing(self,img,Rd,h,K,offset,fx,fno):
        cv2.putText(img,str(fno),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
        self.img=self.draw_frame(img,Rd,h,K,offset,fx)
    def drawing2(self,elps1,elps2,img,Rd,Yd,h,K,offset,fx,fno,logging):
        cv2.putText(img,'Marker found',(10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
        cv2.putText(img,str(fno),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv2.LINE_AA)
        cv2.ellipse(img, elps1, (0, 255, 255), 2)
        cv2.ellipse(img, elps2, (255, 255, 0), 2)
        img=self.draw_frame(img,Rd,h,K,offset,fx)
        self.img=self.draw_frame_2(img,Rd,Yd,h,K,offset,fx)
        if logging is True:
            log.log_detected(fno,self.img)
    def show(self):
        if self.img is not None:
            cv2.imshow('img',cv2.resize(self.img,None,fx=1,fy=1))


class Search:
    def __init__(self):
        self.marker_found=None
        self.elps1=None
        self.elps2=None
    def search(self,l,cont,cnts,hier,marker_found):
        for i in range(len(cont)):
            h0 = hier[0, l+i, :]
            if h0[2]!=-1 and cv2.contourArea(cnts[l+i])>Area1 and cv2.arcLength(cnts[l+i],True)!=0 and ((4*math.pi*cv2.contourArea(cnts[l+i]))/(cv2.arcLength(cnts[l+i],True))**2)>ccl:
                h1 = hier[0, h0[2], :]
#                 print('outer',l+i)
                if h1[2]!=-1 and h1[3]==l+i and h1[0]==-1 and cv2.contourArea(cnts[h0[2]])>Area2 and cv2.arcLength(cnts[h0[2]],True)!=0 and ((4*math.pi*cv2.contourArea(cnts[h0[2]]))/(cv2.arcLength(cnts[h0[2]],True))**2)>ccl:
                    h2 = hier[0, h1[2], :]
#                     print('inner',h0[2])
                    if h2[3]==h0[2] and h2[0]==-1 and h2[2]==-1:
                        self.elps1 = cv2.fitEllipse(cnts[l+i])
                        self.elps2 = cv2.fitEllipse(cnts[h0[2]])
                        self.marker_found=findH(gray_thresh,cnts,h0[2],marker_found)
                        break
    def search_results(self):
        return self.marker_found,self.elps1,self.elps2
    def clear_results(self):
        self.marker_found=False

class Camera_stream:
    def __init__(self):
        #self.control_port  = portname
#         self.cap    = cv2.VideoCapture(0)
        self.cap = WebcamVideoStream(src=0).start()
        self.current_frame = None
        self.active = True
        self.thread = threading.Thread(target=self.get_image, args=())
#         self.thread.daemon=True
        self.thread.start()
    def get_image(self):
        while self.active:
            self.current_frame = self.cap.read()
    def read_image(self):
        return self.current_frame
    def close(self):
        self.active=False
#         self.cap.release()
        self.cap.stop()
        self.thread.join()
        print('Camera_stream stopped')
    
class UAV_stream:
    def __init__(self,portname):
        self.mav_stream = mavutil.mavlink_connection(portname)
        self.mav_stream.mav.command_long_send(
            self.mav_stream.target_system,
            self.mav_stream.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,40000,0,0,0,0,0)
        self.msg=None
        self.active=True
        self.mavlink_thread = threading.Thread(target=self.mavlink_grabber,args=())
#         self.mavlink_thread.daemon=True
        self.mavlink_thread.start()
    def get_data(self):
        while self.active:
            try:
                incoming_msg = self.mav_stream.recv_match().to_dict()
                if incoming_msg["mavpackettype"] == "ATTITUDE":
                    return incoming_msg
            except:
                pass
    def mavlink_grabber(self):
        while self.active:
            self.msg=self.get_data()
    def read(self):
        return self.msg
    def close(self):
        self.active=False
        self.mavlink_thread.join()
        self.mav_stream.mav.command_long_send(
            self.mav_stream.target_system,
            self.mav_stream.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,-1,0,0,0,0,0)
        print('UAV_stream stopped')
        
class logger:
    def __init__(self):
        self.f=open('vision_landing.csv','w',encoding='UTF8',newline='')
        self.writer=csv.writer(self.f)
        header=['datetime','u','v','psi(deg)','theta(deg)','phi(deg)','h','scale','X','Y','fno','exec_time']
        self.writer.writerow(header)
        if not os.path.exists('vision_landing'):
            os.mkdir('vision_landing')
        if not os.path.exists('vision_landing_detected'):
            os.mkdir('vision_landing_detected')
        self.con=180/math.pi
    def log_data(self,date_time,u,v,psi,theta,phi,h,Yd0,Yd1,Yd2,fno,exec_time):
        row=[date_time,u,v,self.con*psi,self.con*theta,self.con*phi,h,Yd0,Yd1,Yd2,fno,exec_time]
        self.writer.writerow(row)
    def log_image(self,fno,img):
        cv2.imwrite('vision_landing/frame'+str(fno)+'.png',img)
    def log_detected(self,fno,img):
        cv2.imwrite('vision_landing_detected/frame'+str(fno)+'.png',img)
    def close(self):
        self.f.close()
        
class plotter:
    def __init__(self):
        self.realX=collections.deque(np.zeros(50))
        self.realY=collections.deque(np.zeros(50))
        self.realZ=collections.deque(np.zeros(50))
        self.realP=collections.deque(np.zeros(50))
        self.realQ=collections.deque(np.zeros(50))
        self.X=np.linspace(0,50,50)
        self.Xn=self.realX
        self.Yn=self.realY
        self.Zn=self.realZ
        self.Pn=self.realP
        self.Qn=self.realQ
        self.active=True
    def update(self,roll,pitch,yaw,Yi):
        self.realX.popleft()
        self.realX.append(con*roll)
        self.realY.popleft()
        self.realY.append(con*pitch)
        self.realZ.popleft()
        self.realZ.append(con*yaw)
        self.realP.popleft()
        self.realP.append(Yi[0])
        self.realQ.popleft()
        self.realQ.append(Yi[1])
        self.Xn=self.realX
        self.Yn=self.realY
        self.Zn=self.realZ
        self.Pn=self.realP
        self.Qn=self.realQ
    def plot(self):
        while self.active:
            plt.subplot(1,5,1)
            plt.cla()
            plt.plot(self.X, self.Xn)
            plt.title("roll")
            plt.grid()
            plt.ylim(-180,180)
            plt.subplot(1,5,2)
            plt.cla()
            plt.plot(self.X, self.Yn)
            plt.title("pitch")
            plt.grid()
            plt.ylim(-180,180)
            plt.subplot(1,5,3)
            plt.cla()
            plt.plot(self.X, self.Zn)
            plt.title("yaw")
            plt.grid()
            plt.ylim(-180,180)
            plt.subplot(1,5,4)
            plt.cla()
            plt.plot(self.X, self.Pn)
            plt.title("Yx")
            plt.grid()
            plt.ylim(-0.3,0.3)
            plt.subplot(1,5,5)
            plt.cla()
            plt.plot(self.X, self.Qn)
            plt.title("Yy")
            plt.grid()
            plt.ylim(-0.3,0.3)
            if plt.waitforbuttonpress(timeout=0.05) is None:
                pass# Wait for user input to continue.
        plt.close()
        print('plot thread stopped')
    def close(self):
        self.active=False


warnings.filterwarnings('ignore')
#set communication with pixhawk
#master = mavutils.mavlink_connection('/dev/ttyACM0',baud=9600)

stream=UAV_stream('/dev/ttyACM0')
#set videostream
#vs=WebcamVideoStream(src=0).start()
cam=Camera_stream()

S=Search()
D=Drawing()
log=logger()
plot=plotter()

if plotting is True:
    plot_thread=threading.Thread(target=plot.plot)
    plot_thread.setDaemon(True)
    plot_thread.start()
    Yi=np.array([0,0],dtype=float)


fno=0
gt=0
while(True):
    start_time=time.perf_counter()
    marker_found=False
    
    msg2,frame=None,None
    msg2=stream.read()
    frame=cam.read_image()
    if frame is None or msg2 is None:
#         print('line 338')
        gt+=time.perf_counter()-start_time
        continue
    
    print('get_time:',gt+(time.perf_counter()-start_time))
    c1=time.perf_counter()
    gt=0
    
    droll=msg2['roll']
    dpitch=msg2['pitch']
    dyaw=msg2['yaw']
    psi=-1*dpitch
    theta=droll
    phi=-1*(dyaw+(math.pi/2))#(use -90 for NED frame representation)
    h=1.25#hs
    
    c2=time.perf_counter()
    
    if logging is True:
        log_image_thread=threading.Thread(target=log.log_image,args=[fno,frame])
        log_image_thread.start()
    #normal detection (currently used for detection)
        
    c3=time.perf_counter()    
        
    imageo = cv2.resize(frame,dsize=(0,0),fx=fx,fy=fy)#scale down image
    
    c4=time.perf_counter()
    
#     if log is True:
#         cv2.imwrite('vision_landing/frame'+str(fno)+'.png',imageo)

    gray = cv2.cvtColor(imageo, cv2.COLOR_BGR2GRAY) #Convert to grayscale
    
    c5=time.perf_counter()
    
    #define blocksize  and c for adaptive threshold
    s = int(gray.shape[1] / (6))
    if s%2==0:
        s=s+1
    c=h*8#8#10 prev
    gray_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, s, c)
    
    c6=time.perf_counter()
    
    if display is True:
#         cv2.imshow('thresh',cv2.resize(gray_thresh,dsize=(0,0),fx=1,fy=1))
        cv2.imshow('thresh',gray_thresh)
        
    c7=time.perf_counter()
    
    _,cnts, hier= cv2.findContours(gray_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    c8=time.perf_counter()
    
    image = imageo.copy()#make a copy for plotting without disturbing original image
#     print(hier)
    c9=time.perf_counter()
    
#     print(len(cnts))
    tn=4
    n=round(len(cnts)/tn)
#     print('n:',n)
    st_list=[]
    for i in range(tn):
        if i < tn-1:
            st=threading.Thread(target=S.search,args=(i*n,cnts[i*n:(i+1)*n],cnts,hier,marker_found))
        else:
            st=threading.Thread(target=S.search,args=(i*n,cnts[i*n:],cnts,hier,marker_found))
        st.start()
        st_list.append(st)
    for i in st_list:
        i.join()

    marker_found,elps1,elps2 = S.search_results()    
    
    c10=time.perf_counter()
    
    Rd=Rotation_matrix(psi,theta,phi)
    
    c11=time.perf_counter()

    
    if marker_found:
        #get image coordinate
        u=(np.array(elps1[0][0])*(1/fx)+np.array(elps2[0][0])*(1/fx))/2
        v=(np.array(elps1[0][1])*(1/fy)+np.array(elps2[0][1])*(1/fy))/2
        Yd=Localize(u,v,Rd,h,K,offset)
        Ydd=Localize(K[0,2],K[1,2],Rd,h,K,offset)
        Yi=np.array([Yd[1]-Ydd[1],Yd[2]-Ydd[2]],dtype=float)
        exec_time=time.perf_counter()-start_time
        c12=time.perf_counter()
        if display is True or logging is True:
            draw_thread=threading.Thread(target=D.drawing2,args=(elps1,elps2,image,Rd,Yd,h,K,offset,fx,fno,logging))
            draw_thread.start()
            D.show()
        c13=time.perf_counter()
        if logging is True:
            log_data_thread=threading.Thread(target=log.log_data,args=[datetime.now(),u,v,
                                                                       psi,theta,phi,h,Yd[0],Yd[1],Yd[2],fno,exec_time])
            log_data_thread.start()
        c14=time.perf_counter()
        if plotting is True:
            plot_update_thread=threading.Thread(target=plot.update,args=(droll,dpitch,dyaw,Yi))
            plot_update_thread.daemon=True
            plot_update_thread.start()
        c15=time.perf_counter()
    else:
        if display is True:
            draw_thread=threading.Thread(target=D.drawing,args=(image,Rd,h,K,offset,fx,fno))
            draw_thread.start()
            D.show()
        c16=time.perf_counter()
        if plotting is True:
            plot_update_thread=threading.Thread(target=plot.update,args=(droll,dpitch,dyaw,Yi))
            plot_update_thread.daemon=True
            plot_update_thread.start()
            
    c17=time.perf_counter()
#     if display is True:
#         cv2.imshow('img',cv2.resize(image,dsize=(0,0),fx=1,fy=1))
    S.clear_results()
    fno+=1
    print('initial calc(c2-c1):{}\nlog original img(c3-c2):{}\nresize img(c4-c3):{}\ngrayscale(c5-c4):{}\nadaptive thresh(c6-c5):{}\nshow thresh img(c7-c6):{}\nfind contours(c8-c7):{}\ncopy img(c9-c8):{}\nsearch cnts(c10-c9):{}\nrotation matrix(c11-c10):{}'.format(
        c2-c1,c3-c2,c4-c3,c5-c4,c6-c5,c7-c6,c8-c7,c9-c8,c10-c9,c11-c10))
    if marker_found:
        print('localize(c12-c11):{}\ndraw2(c13-c12):{}\nlog data(c14-c13):{}\nplot(c15-c14):{}'.format(c12-c11,c13-c12,c14-c13,c15-c14))
    else:
        print('draw(c16-c11):{}\nplot(c17-c16):{}'.format(c16-c11,c17-c16))
    print('total time:',time.perf_counter()-start_time,'\n')
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
log.close()
stream.close()
cam.close()
plot.close()

