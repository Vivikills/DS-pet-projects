import cv2
import pixellib as px
import tensorflow as tf
from pixellib.instance import instance_segmentation


segm_video = instance_segmentation()
segm_video.load_model('mask_rcnn_coco.h5')
target_classes = segm_video.select_target_classes(person=True)
res = segm_video.segmentImage("people.jpg", show_bboxes=True, segment_target_classes= target_classes, extract_segmented_objects=True,save_extracted_objects=True, output_image_name = 'out.jpg' )
print(res)

