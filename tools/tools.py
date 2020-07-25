import os
from collections import defaultdict
from detectron2.structures import BoxMode
from PIL import Image
import itertools

def output_Files():
    os.makedirs('OUTPUT/', exist_ok=True)
    output_files = {
        'train' : 'OUTPUT/wider_face_train_coco.json',
        'val'   : 'OUTPUT/wider_face_val_coco.json',
        'test'  : 'OUTPUT/wider_face_test_coco.json'
    }
    return output_files

def annotations_f():
  annotations = dict()
  if os.path.exists('WIDER/WIDER_train'): annotations["train"] = 'WIDER/wider_face_split/wider_face_train_bbx_gt.txt' 
  if os.path.exists('WIDER/WIDER_val'): annotations["val"] = 'WIDER/wider_face_split/wider_face_val_bbx_gt.txt'
  if os.path.exists('WIDER/WIDER_test'): annotations["test"] = 'WIDER/wider_face_split/wider_face_test_filelist.txt'

  return annotations

def widerface_annotations(annotations=annotations_f()):
  """Function for reading faces bounding box annotations for WIDER Face dataset
  
  Parameters:
  ----------
  :param dict annotations: dictionary with annotations files path 
                           { "train" : WIDER_train_annotations_path,
                             "val"   : WIDER_val_annotations_path }

  :return annotation_dict: annotation dictionary

  Face annotations:
  ----------------
  file_name
  number_of_bounding_box
  bbox [x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose]

  Mappings between attribute names and label values:
  -------------------------------------------------
  blur:
    clear->0
    normal blur->1
    heavy blur->2

  expression:
    typical expression->0
    exaggerate expression->1

  illumination:
    normal illumination->0
    extreme illumination->1

  occlusion:
    no occlusion->0
    partial occlusion->1
    heavy occlusion->2

  pose:
    typical pose->0
    atypical pose->1

  invalid:
    false->0 (valid image)
    true->1 (invalid image)
  """

  annotation_dict = defaultdict(list)
  for key, value in annotations.items():
    with open(value, "r") as file_:
      rows = file_.readlines()

    idx = 0
    while (idx < len(rows)):
      file_name = rows[idx].replace("\n", "")
      number_of_bounding_box = int(rows[idx+1]) if key !='test' else 0
      bbox = []
      '''
      Attention! there are photos without annotations..
      0--Parade/0_Parade_Parade_0_452.jpg
      0
      0 0 0 0 0 0 0 0 0 0 
      '''
      if key != 'test':
        jump = number_of_bounding_box if number_of_bounding_box != 0 else 1

        for i in range(1, jump+1):
          box = rows[idx+1+i]
          box = [int(item) for item in box.split(' ')[:10]]
          bbox.append(box)

      annotation_dict[key].append({
          'file_name'             : file_name,
          'number_of_bounding_box': number_of_bounding_box, 
          'bbox'                  : bbox if key != 'test' else [[0,0,0,0,0,0,0,0,0,0]]
      })
      idx += (jump+2) if key != 'test' else 1

  return annotation_dict

def PATH_dict(key):
  PATH_dict = { 
      'train'           : 'WIDER/WIDER_train',      # path to the train directory
      'val'             : 'WIDER/WIDER_val',        # path to the validation directory 
      'test'            : 'WIDER/WIDER_test',
      'annotation_dir'  : 'WIDER/wider_face_split'  # path to annotation directory 
  }
  return PATH_dict[key]

def annotations(key): 
  IMAGES_PATH=PATH_dict(key)
  annotation_dict=widerface_annotations()[key]

  dataset_dicts = []
  for idx, item in enumerate(annotation_dict):
    record = {}
    record["image_id"] = idx
    record['file_name'] = os.path.join(os.path.join(IMAGES_PATH,"images"), item["file_name"])
    img_ = Image.open(os.path.join(os.path.join(IMAGES_PATH,"images"), item["file_name"]))
    record['width'] = img_.width
    record['height'] = img_.height
    bbox = [i[:4] for i in item['bbox']]
    annotation = []
    for box in bbox:
        anno = {}
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[0])+int(box[2]), int(box[1])+int(box[3])
        polygon = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        polygon = list(itertools.chain.from_iterable(polygon))
        anno['category_id'] = 0 
        anno['iscrowd'] = 0
        anno['bbox'] = [xmin, ymin, xmax, ymax]
        anno['bbox_mode'] = BoxMode.XYXY_ABS
        anno['segmentation'] = [polygon]
        annotation.append(anno)
    record["annotations"] = annotation    
    dataset_dicts.append(record)
  return dataset_dicts