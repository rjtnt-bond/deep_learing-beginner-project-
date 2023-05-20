import cv2
import pixellib
from pixellib.instance import instance_segmentation
ditact_image = instance_segmentation()
ditact_image.load_model("mask_rcnn_coco.h5") 
runtime_photo = cv2.VideoCapture(0)

while runtime_photo.isOpened():
    res,frame=runtime_photo.read()
   
    result=ditact_image.segmentFrame(frame,show_bboxes=True)
    photo=result[1]
    cv2.imshow('Image Segmentation',photo)

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

runtime_photo.release()
cv2.destroyAllWindows()

