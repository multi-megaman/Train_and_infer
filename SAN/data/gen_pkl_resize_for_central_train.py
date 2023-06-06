import os
import argparse
import glob
from tqdm import tqdm
import cv2
import pickle as pkl
from .gen_hybrid_data_for_central_train import Gen_hybrid

parser = argparse.ArgumentParser(description='Spatial channel attention')
parser.add_argument('--images_path', default='/home/yuanye/work/data/CROHME2014/14_off_image_test', type=str, help='image')
parser.add_argument('--labels_path', default='/home/yuanye/work/data/CROHME2014/test_caption.txt', type=str, help='label')
parser.add_argument('--train_test', default='/home/yuanye/work/data/CROHME2014/test_caption.txt', type=str, help='train/test')
parser.add_argument('--image_type', default='png', type=str, help='png/jpg/bmp/jpeg/etc')
args = parser.parse_args()

def Gen_pkl(images_path,labels_path,train_or_test,image_type):

    
    image_path = images_path #'./train/train_images'
    image_out ='data/SAN/' +"SAN_"+ train_or_test + '_image.pkl'
    # laebl_path = args.labels_path #'./train'
    laebl_path= Gen_hybrid(labels_path=labels_path,train_or_tests=train_or_test)
    label_out ='data/SAN/' +"SAN_"+ train_or_test + '_label.pkl'

    if os.path.exists(label_out):
        os.remove(os.path.abspath(label_out))
    if os.path.exists(image_out):
        os.remove(os.path.abspath(image_out))

    # images = glob.glob(os.path.join(image_path, '*.'+args.image_type))
    image_dict = {}

    # for item in tqdm(images):
    #
    #     img = cv2.imread(item)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     dim = (150, 150)
    #     img = cv2.resize(img, (dim), interpolation = cv2.INTER_AREA)
    #     image_dict[os.path.basename(item).replace('.'+args.image_type,'')] = img



    labels = glob.glob(os.path.join(laebl_path, '*.txt'))
    label_dict = {}

    for item in tqdm(labels):
        with open(item,encoding="utf8") as f:
            lines = f.readlines()
        label_dict[os.path.basename(item).replace('.txt','')] = lines

        img_name = os.path.basename(item).replace('txt', image_type)
        img_name = os.path.join(image_path, img_name)

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dim = (150, 150)
        img = cv2.resize(img, (dim), interpolation = cv2.INTER_AREA)
        image_dict[os.path.basename(img_name).replace('.'+image_type,'')] = img



    with open(label_out,'wb') as f:
        pkl.dump(label_dict, f)

    with open(image_out,'wb') as f:
        pkl.dump(image_dict, f)

    return str(os.path.abspath(image_out)), str(os.path.abspath(label_out))