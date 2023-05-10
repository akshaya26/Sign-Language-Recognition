import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import imutils


word_dict = {0:'One',1:'Two',2:'Three',3:'Four',4:'Five',5:'Six',6:'Seven',7:'Eight',8:'Nine', 9:'A', 10: 'B',
             11:'C',12:'D',13:'E',14:'F',15:'G',16:'H',17:'I',18:'J', 19:'Karan', 20: 'L',
             21:'M',22:'N',23:'O',24:'P',25:'Q',26:'R',27:'S',28:'T', 29:'U', 30: 'V',
             31:'W',32:'X',33:'Y',34:'Z'}

model = keras.models.load_model("./best.h5")

background = None
accumulated_weight = 1

ROI_top = 136
ROI_bottom = 264
ROI_right = 186
ROI_left = 314

print('A')
def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=30):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand
        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)

print('B')
cam = cv2.VideoCapture(0)
# v=cam.get(cv2.cv2.CV_CAP_PROP_FPS)
# print(v)
print('C')
num_frames = 0
while True:
    ret, frame = cam.read()
    cv2.waitKey(150)

    # filpping the frame to prevent inverted image of captured frame...
    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    # ROI from the frame
    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (3,3), 0)

    if num_frames < 70:

        cal_accum_avg(gray_frame, accumulated_weight)

        cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)

    else:
        # segmenting the hand region
        hand = segment_hand(gray_frame)

        # Checking if we are able to detect the hand...
        if hand is not None:
            thresholded, hand_segment = hand

            # Drawing contours around hand segment
            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

            
            cv2.imshow("Thesholded Hand Image", thresholded)
            # thresholded = cv2.resize(thresholded, (64, 64))
            # thresholded = tf.image.resize_with_pad(thresholded, 64, 64, method=ResizeMethod.BILINEAR, antialias=False)
            r=64.0/thresholded.shape[0]
            dim=(int(thresholded.shape[1]*r),64)
            thresholded=cv2.resize(thresholded,dim,interpolation=cv2.INTER_AREA)

            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            cv2.imwrite("./newimg.jpg",thresholded)
            thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))
           
            pred = model.predict(thresholded)
            # print("pred..........",pred)
            to_speech = word_dict[np.argmax(pred)]
            print(type(pred),np.argmax(pred),to_speech,'\n')
            if(pred[0][np.argmax(pred)]>=0.8):
                cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif(to_speech=="One" or to_speech=="Six"):
                model_1_6 = keras.models.load_model("./best_1_6.h5")
                pred = model.predict(thresholded)
                # print("pred..........",pred)
                to_speech = word_dict[np.argmax(pred)]
                print("In 1/6", type(pred),np.argmax(pred),to_speech,'\n')
              #  add new model                
                cv2.putText(frame_copy, word_dict[np.argmax(pred)] , (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                cv2.putText(frame_copy, "NONE", (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # Draw ROI on frame_copy
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)

    # incrementing the number of frames for tracking
    num_frames += 1

    # Display the frame with segmented hand
    cv2.putText(frame_copy, "Value_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    cv2.imshow("Sign Detection", frame_copy)

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

# Release the camera and destroy all the windows
cam.release()
cv2.destroyAllWindows()
