# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import cStringIO
import sys
import tempfile
import math 

MODEL_BASE = '/home/itaizloto/Downloads/models'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')

sys.path.append(MODEL_BASE + '/object_detection/utils')

from decorator import requires_auth
from flask import Flask
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from flask_wtf.file import FileField
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
import scipy.ndimage
from PIL import Image,ImageDraw,ImageFilter,ImageOps,ImageEnhance
#from google.cloud import storage


app = Flask(__name__)


@app.before_request
@requires_auth
def before_request():
  pass


PATH_TO_CKPT = MODEL_BASE + '/object_detection/pathology_graph_faster_rcnn_final/frozen_inference_graph.pb'
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/object-detection.pbtxt'

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())

ANGLE = 90
BRIGHTNESS = 5

def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image


class PhotoForm(Form):
  input_photo = FileField(
      'File extension should be: %s (case-insensitive)' % ', '.join(extensions),
      validators=[is_image()])


class ObjectDetector(object):

  def __init__(self):
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections


def draw_bounding_box_on_image(image, box, color='red', thickness=2):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def encode_image(image):
  image_buffer = cStringIO.StringIO()
  image.save(image_buffer, format='PNG')
  imgstr = 'data:image/png;base64,{:s}'.format(
      base64.b64encode(image_buffer.getvalue()))
  return imgstr

def load_image(image_path):
  image = Image.open(image_path).convert('RGB')
  return image
  
def split_image_to_corners(image):
  corner_left = crop_image_to_corner(image,True)
  corner_right = crop_image_to_corner(image,False)
  return corner_left, corner_right
  
def find_best_augmentation(image, num_results,brightness_span=1,angle=90):
  best_score = 0
  best_scores = []
  best_boxes =[]
  best_classes = []
  best_num_detections = 0
  best_rotation_angle = 0
  best_image_brightness = 0
  max_current_score = 0
  contrast = ImageEnhance.Contrast(image)
  
  for val_bright in range(10-brightness_span,10+brightness_span+1):
    for val_rotate in range(0,360,angle):
      if val_rotate != 0:
        image_brighter = contrast.enhance(val_bright)
        image_rotated = image_brighter.rotate(val_rotate)
        boxes, scores, classes, num_detections = client.detect(image_rotated)
        max_current_score = np.max(scores)
        if max_current_score > best_score:
          best_score = max_current_score
          best_boxes = boxes
          best_scores = scores
          best_classes = classes
          best_num_detections = num_detections
          best_rotation_angle = val_rotate
          best_image_brightness = val_bright
  return best_boxes, best_scores, best_classes, best_num_detections, best_rotation_angle, best_image_brightness
  
def convert_box_to_original(im_width, im_height, best_boxes, best_rotation_angle):
  ymin, xmin, ymax, xmax = best_boxes[0]
  image_center_point = [0.5,0.5]
  new_lb_point = point_rotate(image_center_point,[ymin,xmin],math.radians(best_rotation_angle))
  new_rt_point = point_rotate(image_center_point,[ymax,xmax],math.radians(best_rotation_angle))
  new_lt_point = point_rotate(image_center_point,[ymax,xmin],math.radians(best_rotation_angle))
  new_rb_point = point_rotate(image_center_point,[ymin,xmax],math.radians(best_rotation_angle))
  xmin_rotated = np.min([new_lb_point[1],new_rb_point[1],new_lt_point[1],new_rt_point[1]])
  ymin_rotated = np.min([new_lb_point[0],new_rb_point[0],new_lt_point[0],new_rt_point[0]])
  xmax_rotated = np.max([new_lb_point[1],new_rb_point[1],new_lt_point[1],new_rt_point[1]])
  ymax_rotated = np.max([new_lb_point[0],new_rb_point[0],new_lt_point[0],new_rt_point[0]])
  return [ymin_rotated,xmin_rotated,ymax_rotated,xmax_rotated]
  
# helper functions
def crop_image_to_corner(original,isLeftSide=True):
  (width, height) = original.size   # Get dimensions
  left = 0
  right = 0
  top = 0
  bottom = 0
  if isLeftSide:
    left = 0
    top = height/2
    right = width/4
    bottom = height
  else: # right side
    left = 3 * width/4
    top = height/2
    right = width
    bottom = height
  cropped_image = original.crop((left, top, right, bottom))
  return cropped_image 

def corner_to_panoramic(rotated_box_points,isLeftSide=True):
  ymi, xmi, yma, xma = rotated_box_points
  ratio = 0.75 #3/4
  if isLeftSide:
    ratio = 0 #1/4
  else:
    ymin = 0.5 + ymi/2
    xmin = ratio + xmi/4
    ymax = 0.5 + yma/2
    xmax = ratio + xma/4
  return ymin,xmin,ymax,xmax

def rotate_image(cover, val):
  rotated_img = imutils.rotate_bound(cover, val)
  return rotated_img
  
def point_rotate(origin, point, angle):
  """
  Rotate a point counterclockwise by a given angle around a given origin.
  The angle should be given in radians.
  """
  oy, ox = origin
  py, px = point

  qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
  qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
  return qy, qx
  
# main  
def detect_objects(image_path):
  image  = load_image(image_path)
  corner_left_img, corner_right_img = split_image_to_corners(image)
  left_img_best_boxes, left_img_best_scores, left_img_best_classes, left_img_best_num_detections, left_img_best_rotation_angle, left_img_best_image_brightness = find_best_augmentation(corner_left_img,0,BRIGHTNESS, ANGLE)
  right_img_best_boxes, right_img_best_scores, right_img_best_classes, right_img_best_num_detections, right_img_best_rotation_angle, right_img_best_image_brightness = find_best_augmentation(corner_right_img,0,BRIGHTNESS, ANGLE)
  cropped_image = corner_left_img
  best_boxes = left_img_best_boxes
  best_scores = left_img_best_scores
  best_classes = left_img_best_classes
  best_num_detections = left_img_best_num_detections
  best_rotation_angle = left_img_best_rotation_angle
  best_image_brightness = left_img_best_image_brightness
  isRightCornerGotHigherScore = left_img_best_scores[0] < right_img_best_scores[0]
  if isRightCornerGotHigherScore:
    cropped_image = corner_right_img
    best_boxes = right_img_best_boxes
    best_scores = right_img_best_scores
    best_classes = right_img_best_classes
    best_num_detections = right_img_best_num_detections
    best_rotation_angle = right_img_best_rotation_angle
  max_score = best_scores[0]
  image.thumbnail((480, 480), Image.ANTIALIAS)
  new_images = {}
  (im_width, im_height) = cropped_image.size
  rotated_box_points = convert_box_to_original(im_width, im_height, best_boxes,best_rotation_angle)
  cls = best_classes[0]
  if cls not in new_images.keys():
    new_images[cls] = image.copy()

  ymin,xmin,ymax,xmax = corner_to_panoramic(rotated_box_points,isRightCornerGotHigherScore==False)
  draw_bounding_box_on_image(new_images[cls], [ymin,xmin,ymax,xmax])
  result = {}
  result['original'] = encode_image(image.copy())
  result['score'] = [best_image_brightness, best_rotation_angle, round(max_score*100,3)]
  for cls, new_image in new_images.iteritems():
    category = client.category_index[cls]['name']
    result[category] = encode_image(new_image)
  return result
  

def get_boxed_image(image):
  cropped_image = crop_image_to_corner(image)
  boxes, scores, classes, num_detections = client.detect(cropped_image)
  image.thumbnail((480, 480), Image.ANTIALIAS)
  new_images = {}
  max_score = np.max(scores)
  for i in range(num_detections):
    if scores[i] < 0.5: continue
    cls = classes[i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    ymi, xmi, yma, xma = boxes[i]
    ymin = 0.5 + ymi/2
    xmin = xmi/4
    ymax = 0.5 + yma/2
    xmax = xma/4
    draw_bounding_box_on_image(new_images[cls], [ymin,xmin,ymax,xmax])
  return new_images, max_score

@app.route('/')
def upload():
  #uploaded_file = request.files.get('file')

  #gcs = storage.Client(project='primordial-ship-89113')
  #bucket = gcs.get_bucket('my_data_path')
  #blob = bucket.blob(uploaded_file.filename)

  #blob.upload_from_string(
  #  uploaded_file.read(),
  #  content_type=uploaded_file.content_type
  #)
  photo_form = PhotoForm(request.form)
  return render_template('upload.html', photo_form=photo_form, result={})


@app.route('/post', methods=['GET', 'POST'])
def post():
  form = PhotoForm(CombinedMultiDict((request.files, request.form)))

  if request.method == 'POST' and form.validate():
    with tempfile.NamedTemporaryFile() as temp:
      form.input_photo.data.save(temp)
      temp.flush()
      result = detect_objects(temp.name)

    photo_form = PhotoForm(request.form)
    return render_template('upload.html',
                           photo_form=photo_form, result=result)
  else:
    return redirect(url_for('upload'))


client = ObjectDetector()


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=80, debug=False)