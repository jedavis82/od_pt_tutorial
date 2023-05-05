"""
The Fruits dataset comes as only a train and test set. All annotations are also in XML format.
This script will create a train/test/val split and then create an annotations CSV file instead of
having to load each XML file when using the Torch Dataloader class.
"""
import os
import shutil
import numpy as np
import xmltodict
import pandas as pd
import cv2


data_dir = '../data/fruits/'
train_dir = f'{data_dir}train/'
val_dir = f'{data_dir}val/'
test_dir = f'{data_dir}test/'
temp_dir = f'{data_dir}temp/'
train_ann_file = f'{train_dir}annotations.csv'
val_ann_file = f'{val_dir}annotations.csv'
test_ann_file = f'{test_dir}annotations.csv'


def move_to_temp_dir():
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)

    train_files = [f'{train_dir}{f}' for f in train_files]
    test_files = [f'{test_dir}{f}' for f in test_files]

    train_files_mv = [f'{temp_dir}{f}' for f in os.listdir(train_dir)]
    test_files_mv = [f'{temp_dir}{f}' for f in os.listdir(test_dir)]
    train_moved = [shutil.move(curr, move) for curr, move in zip(train_files, train_files_mv)]
    test_moved = [shutil.move(curr, move) for curr, move in zip(test_files, test_files_mv)]


def move_files(source_dir, dest_dir, files):
    source_files_jpg = [f'{source_dir}{f}.jpg' for f in files]
    dest_files_jpg = [f'{dest_dir}{f}.jpg' for f in files]
    source_files_xml = [f'{source_dir}{f}.xml' for f in files]
    dest_files_xml = [f'{dest_dir}{f}.xml' for f in files]
    # Move the .jpg files
    move_jpg = [shutil.move(curr, mv) for curr, mv in zip(source_files_jpg, dest_files_jpg)]
    # Move the .xml files
    move_xml = [shutil.move(curr, mv) for curr, mv in zip(source_files_xml, dest_files_xml)]


def create_splits():
    temp_files = os.listdir(temp_dir)

    # Get the file names without the extension
    temp_names = [os.path.splitext(f)[0] for f in temp_files if f.lower().endswith('.jpg')]
    # Create an array of each type of fruit image
    apple_files = [f for f in temp_names if 'apple' in f]
    banana_files = [f for f in temp_names if 'banana' in f]
    orange_files = [f for f in temp_names if 'orange' in f]
    mixed_files = [f for f in temp_names if 'mixed' in f]

    # Create a split of the data using numpy
    ap_len = len(apple_files)
    ba_len = len(banana_files)
    or_len = len(orange_files)
    mi_len = len(mixed_files)
    splits = [0.8, 0.1, 0.1]
    train_r, test_r, val_r = splits
    ap_split_indices = [int(ap_len*train_r), int(ap_len*(train_r+val_r))]
    ap_train, ap_val, ap_test = np.split(apple_files, ap_split_indices)
    ba_split_indices = [int(ba_len * train_r), int(ba_len * (train_r + val_r))]
    ba_train, ba_val, ba_test = np.split(banana_files, ba_split_indices)
    or_split_indices = [int(or_len * train_r), int(or_len * (train_r + val_r))]
    or_train, or_val, or_test = np.split(orange_files, or_split_indices)
    mi_split_indices = [int(mi_len * train_r), int(mi_len * (train_r + val_r))]
    mi_train, mi_val, mi_test = np.split(mixed_files, mi_split_indices)

    apple_train = ap_train.tolist()
    apple_val = ap_val.tolist()
    apple_test = ap_test.tolist()
    banana_train = ba_train.tolist()
    banana_val = ba_val.tolist()
    banana_test = ba_test.tolist()
    orange_train = or_train.tolist()
    orange_val = or_val.tolist()
    orange_test = or_test.tolist()
    mixed_train = mi_train.tolist()
    mixed_val = mi_val.tolist()
    mixed_test = mi_test.tolist()

    # Move our splits to the corresponding directories
    move_files(temp_dir, train_dir, apple_train)
    move_files(temp_dir, val_dir, apple_val)
    move_files(temp_dir, test_dir, apple_test)
    move_files(temp_dir, train_dir, banana_train)
    move_files(temp_dir, val_dir, banana_val)
    move_files(temp_dir, test_dir, banana_test)
    move_files(temp_dir, train_dir, orange_train)
    move_files(temp_dir, val_dir, orange_val)
    move_files(temp_dir, test_dir, orange_test)
    move_files(temp_dir, train_dir, mixed_train)
    move_files(temp_dir, val_dir, mixed_val)
    move_files(temp_dir, test_dir, mixed_test)


def parse_xml_files(input_dir):
    # Store an array of dictionaries to return. These will be used to create a dataframe
    results = []
    filepaths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith('.xml')]
    for f in filepaths:
        with open(f, encoding='utf-8') as xml:
            data = xmltodict.parse(xml.read())
        # Grab the fields we wish to store from the data dictionary
        data_dict = data['annotation']
        fname = data_dict['filename']
        fruits = data_dict['object']

        # Use OpenCV to get the image width and height to store in the dataframe
        img_path = f'{input_dir}{fname}'
        img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
        img_width, img_height = img.shape[0], img.shape[1]

        # There are two scenarios here for results:
        # 1. There is only one fruit in the image, in which case type(fruits) is a dict object
        # 2. There are multiple fruits in the image, in which case type(fruits) is a list
        # We will handle both cases
        if type(fruits) is dict:
            # only one fruit
            res = {'filename': fname, 'label': fruits['name'], 'img_width': img_width, 'img_height': img_height}
            bndbox = fruits['bndbox']  # Extract the bounding box from the xml dict
            x0, y0, x1, y1 = int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])
            res['x0'] = x0
            res['y0'] = y0
            res['x1'] = x1
            res['y1'] = y1
            results.append(res)
        elif type(fruits) is list:
            # Multiple fruits
            for fruit in fruits:
                res = {'filename': fname, 'label': fruit['name'], 'img_width': img_width, 'img_height': img_height}
                bndbox = fruit['bndbox']
                x0, y0, x1, y1 = int(bndbox['xmin']), int(bndbox['ymin']), int(bndbox['xmax']), int(bndbox['ymax'])
                res['x0'] = x0
                res['y0'] = y0
                res['x1'] = x1
                res['y1'] = y1
                results.append(res)
    results_df = pd.DataFrame.from_dict(results)
    return results_df


def main():
    # Verify that the data_dir exists
    assert os.path.exists(data_dir), "Data Directory does not exist"

    # First, we'll move all the image and annotation files to a temporary directory for creating out splits
    # If the temp directory exists, we have already moved our images over.
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        move_to_temp_dir()
    # Now we'll create the splits and move the files
    # If the val directory exists we've already created the splits
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        create_splits()
    # Now we will create CSV annotation files from the XML files shipped with the data set
    train_df = parse_xml_files(train_dir)
    val_df = parse_xml_files(val_dir)
    test_df = parse_xml_files(test_dir)

    train_df.to_csv(train_ann_file, encoding='utf-8', header=True, index=False)
    val_df.to_csv(val_ann_file, encoding='utf-8', header=True, index=False)
    test_df.to_csv(test_ann_file, encoding='utf-8', header=True, index=False)
    print()


if __name__ == '__main__':
    main()
