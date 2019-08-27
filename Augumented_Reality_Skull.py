# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

upper_skull = cv2.imread('upper_skull.jpg')
lower_skull = cv2.imread('lower_skull.jpg')
#rows,cols,ch = Skull.shape

cap = cv2.VideoCapture(0)

#count = 0
#is_shivam = False

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ret, frame2 = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        #print(shape)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        #Left Eyebrow

        Left_Eyebro_18_x = float(shape[17][0])
        Left_Eyebro_18_y = float(shape[17][1])

        Left_Eyebro_22_x = float(shape[21][0])
        Left_Eyebro_22_y = float(shape[21][1])

        #Right Eyebrow

        Right_Eyebro_23_x = float(shape[22][0])
        Right_Eyebro_23_y = float(shape[22][1])

        Right_Eyebro_27_x = float(shape[26][0])
        Right_Eyebro_27_y = float(shape[26][1])

        #Left Eye

        Left_Eye_37_x = float(shape[36][0])
        Left_Eye_37_y = float(shape[36][1])

        Left_Eye_40_x = float(shape[39][0])
        Left_Eye_40_y = float(shape[39][1])


        #Right Eye

        Right_Eye_43_x = float(shape[42][0])
        Right_Eye_43_y = float(shape[42][1])

        Right_Eye_46_x = float(shape[45][0])
        Right_Eye_46_y = float(shape[45][1])

        #Nose

        Nose_28_x = float(shape[27][0])
        Nose_28_y = float(shape[27][1])

        Nose_31_x = float(shape[30][0])
        Nose_31_y = float(shape[30][1])

        # Nose_Lower

        Nose_Lower_32_x = float(shape[31][0])
        Nose_Lower_32_y = float(shape[31][1])

        Nose_Lower_36_x = float(shape[35][0])
        Nose_Lower_36_y = float(shape[35][1])

        #Nose_Tip

        Nose_Tip_31_x = float(shape[32][0])
        Nose_Tip_31_y = float(shape[32][1])

        #Chin

        Chin_9_x = float(shape[8][0])
        Chin_9_y = float(shape[8][1])

        #Lips

        Lips_49_x = float(shape[48][0])
        Lips_49_y = float(shape[48][1])

        Lips_55_x = float(shape[54][0])
        Lips_55_y = float(shape[54][1])


        Lips_61_x = float(shape[60][0])
        Lips_61_y = float(shape[60][1])

        Lips_62_x = float(shape[61][0])
        Lips_62_y = float(shape[61][1])

        Lips_63_x = float(shape[62][0])
        Lips_63_y = float(shape[62][1])

        Lips_64_x = float(shape[63][0])
        Lips_64_y = float(shape[63][1])

        Lips_65_x = float(shape[64][0])
        Lips_65_y = float(shape[64][1])

        Lips_68_x = float(shape[67][0])
        Lips_68_y = float(shape[67][1])

        Lips_67_x = float(shape[66][0])
        Lips_67_y = float(shape[66][1])

        Lips_66_x = float(shape[65][0])
        Lips_66_y = float(shape[65][1])



        #Face Horizontal

        Face_Horizontal_1_x = float(shape[0][0])
        Face_Horizontal_1_y = float(shape[0][1])

        Face_Horizontal_17_x = float(shape[16][0])
        Face_Horizontal_17_y = float(shape[16][1])

        #Distance

        #Distance_Left_Eyebro = math.hypot((Left_Eyebro_18_x-Left_Eyebro_22_x),(Left_Eyebro_18_y-Left_Eyebro_22_y))
        #Distance_Right_Eyebro = math.hypot((Right_Eyebro_23_x - Right_Eyebro_27_x), (Right_Eyebro_23_y - Right_Eyebro_27_y))
        #Distance_Left_Eye = math.hypot((Left_Eye_37_x - Left_Eye_40_x),(Left_Eye_37_y - Left_Eye_40_y))
        #Distance_Right_Eye = math.hypot((Right_Eye_43_x - Right_Eye_46_x),(Right_Eye_43_y - Right_Eye_46_y))
        #Distance_Nose = math.hypot((Nose_28_x - Nose_31_x), (Nose_28_y - Nose_31_y))
        #Distance_Nose_Lower = math.hypot((Nose_Lower_32_x - Nose_Lower_36_x), (Nose_Lower_32_y - Nose_Lower_36_y))
        #Distance_Lips = math.hypot((Lips_49_x - Lips_55_x), (Lips_49_y - Lips_55_y))
        #Distance_NoseTip_Chin = math.hypot((Nose_Tip_31_x - Chin_9_x), (Nose_Tip_31_y - Chin_9_y))
        #Distance_Face_Horizontal = math.hypot((Face_Horizontal_1_x - Face_Horizontal_17_x), (Face_Horizontal_1_y - Face_Horizontal_17_y))

        #if (Face_Horizontal_1_x != None):


        #Resizing Upper_Skull According to face points
        pts1_lower = np.float32([[1,12], [349, 17], [181, 141]])
        pts2_lower = np.float32([[Face_Horizontal_1_x,Face_Horizontal_1_y], [Face_Horizontal_17_x,Face_Horizontal_17_y], [Lips_67_x,Lips_67_y]])

        M_lower = cv2.getAffineTransform(pts1_lower, pts2_lower)

        dst_lower = cv2.warpAffine(lower_skull, M_lower, (640,480))

        rows_lower, cols_lower, channels = dst_lower.shape
        roi_lower = frame2[0:rows_lower, 0:cols_lower]

        # Now create a mask of logo and create its inverse mask also
        dst_gray_lower = cv2.cvtColor(dst_lower, cv2.COLOR_BGR2GRAY)
        ret_lower, mask_lower = cv2.threshold(dst_gray_lower, 6, 255, cv2.THRESH_BINARY)

        #cv2.imshow('mask', mask)
        mask_inv_lower = cv2.bitwise_not(mask_lower)
        #cv2.imshow('not_mask', mask_inv)

        # Now black-out the area of logo in ROI
        frame2_bg = cv2.bitwise_and(roi_lower, roi_lower, mask=mask_inv_lower)

        # Take only region of logo from logo image.
        dst_fg_lower = cv2.bitwise_and(dst_lower, dst_lower, mask=mask_lower)

        # Put logo in ROI and modify the main image
        dst2 = cv2.add(frame2_bg, dst_fg_lower)
        frame2[0:rows_lower, 0:cols_lower] = dst2




        #Resizing Upper_Skull According to face points
        pts1_upper = np.float32([[21,299], [368, 299], [201, 448]])
        pts2_upper = np.float32([[Face_Horizontal_1_x,Face_Horizontal_1_y], [Face_Horizontal_17_x,Face_Horizontal_17_y], [Lips_63_x,Lips_63_y]])

        M_upper = cv2.getAffineTransform(pts1_upper, pts2_upper)

        dst_upper = cv2.warpAffine(upper_skull, M_upper, (640,480))

        rows_upper, cols_upper, channels = dst_upper.shape
        roi_upper = frame2[0:rows_upper, 0:cols_upper]

        # Now create a mask of logo and create its inverse mask also
        dst_gray_upper = cv2.cvtColor(dst_upper, cv2.COLOR_BGR2GRAY)
        ret_upper, mask_upper = cv2.threshold(dst_gray_upper, 6, 255, cv2.THRESH_BINARY)

        #cv2.imshow('mask', mask)
        mask_inv_upper = cv2.bitwise_not(mask_upper)
        #cv2.imshow('not_mask', mask_inv)

        # Now black-out the area of logo in ROI
        frame2_bg = cv2.bitwise_and(roi_upper, roi_upper, mask=mask_inv_upper)

        # Take only region of logo from logo image.
        dst_fg_upper = cv2.bitwise_and(dst_upper, dst_upper, mask=mask_upper)

        # Put logo in ROI and modify the main image
        dst2 = cv2.add(frame2_bg, dst_fg_upper)
        frame2[0:rows_upper, 0:cols_upper] = dst2


        cv2.imshow("dst_lower",dst_lower)
        cv2.imshow("dst_upper", dst_upper)
        cv2.imshow("dst2", dst2)

    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
