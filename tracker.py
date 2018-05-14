import numpy as np
import cv2
cap = cv2.VideoCapture(0)# video from the webcam
# shi tomasi corner detection
corner_point = dict( maxCorners = 1,qualityLevel = 0.3,minDistance = 7,useHarrisDetector=True,blockSize = 7 )

## lucas kanade optical flow
lucas_kanade = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

## to select any random color to draw on the figure
color = np.random.randint(0,255,(100,3))
## select the good features from the first frame
ret, frame1 = cap.read()
o_g = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(o_g, mask = None, **corner_point)

mask = np.zeros_like(frame1)## create the mask image 
while(1):
    ret,frame = cap.read()
    f_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    p1, st, err = cv2.calcOpticalFlowPyrLK(o_g, f_g, p0, None, **lucas_kanade)

    g_n = p1[st==1]
    g_o = p0[st==1]

    for i,(new,old) in enumerate(zip(g_n,g_o)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imshow('Test',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    ### update both the frame and the points
    o_g = f_g.copy()
    p0 = g_n.reshape(-1,1,2)
cv2.destroyAllWindows()
cap.release()