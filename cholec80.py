from torch.utils.data.dataset import Dataset
from bot import *

from PIL import Image, ImageFile

import re
import os
import sys
import datetime

FRAMES_PER_IMAGE = 25

# video file structure: 1 image per 25 framesn
#  image_path: /root_path/[set_number]/[video_number]/[image_number].png
#  video_dir:  /root_path/[set_number]/[video_number]/
#  set_dir:    /root_path/[set_number]/
#  root_dir:   /root_path/
error_path = '/home/justaviju/PycharmProjects/resnet/errormsgs/'

idx_label = {
    'preparation': 0, 
    'calottriangledissection': 1, 
    'cleaningcoagulation': 2, 
    'gallbladderdissection': 3, 
    'gallbladderretraction': 4, 
    'clippingcutting': 5, 
    'gallbladderpackaging': 6
}


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_labels(video_dir, annotations_dir):
    """Finds all labels from a given video path and a annotations path"""

    video_number = video_dir[-2:]
    annotations_file = 'video{}-phase.txt'.format(video_number)
    label_by_first_frame = {}
    labels_set = set()
    is_data_line = False

    with open(os.path.join(annotations_dir,annotations_file), 'r') as annotation_file:
        for annotation_line in annotation_file:
            annotation_line = annotation_line.lower()
            frame_num = re.sub("\D", "", annotation_line)
            label = re.sub("\d+\s+", "", annotation_line).strip()

            if label not in labels_set and is_data_line:
                labels_set.add(label)
                label_by_first_frame[int(frame_num)] = label
            else:
                is_data_line = True
    return label_by_first_frame


def get_label(image_path, labels_sorted_by_first_frame, idx_by_label):
    """Returns the label for a single image"""

    frame_image_name = re.sub(".*\/", "", image_path)
    image_number = int(re.sub("\D+$", "", frame_image_name))
    frame_number = image_number * FRAMES_PER_IMAGE
    # initialize with first frame
    previous_label = labels_sorted_by_first_frame[0][1]

    for first_frame_num, label in labels_sorted_by_first_frame:
        if frame_number < first_frame_num:
            return image_path, idx_by_label[previous_label]

        previous_label = label

    last_label = labels_sorted_by_first_frame[-1][1]

    return image_path, idx_by_label[last_label]


def make_dataset(root_dir, annotations_dir, image_file_extensions):
    progress = '#'
    fill = '-'
    images = []
    labels = []
    idx_by_label = []

    for current_dir, dir_names, file_names in sorted(os.walk(root_dir)):
        if dir_names:
            current = 0
            total = len(dir_names)
            print("\n", current_dir)
            continue

        current += 1

        if total:
            fraction = float(current)/total
            percent = round(fraction * total)
            sys.stdout.write("\r[{}{}]{:.2f}%".format((progress * percent), fill* (total - percent),(fraction * 100)))
        label_by_first_frame = find_labels(current_dir, annotations_dir)
        labels_sorted_by_first_frame = sorted(label_by_first_frame.items())

        labels = label_by_first_frame.values()
        idx_by_label = get_idx_by_label(labels)

        for image_file_name in file_names:
            if has_file_allowed_extension(image_file_name, image_file_extensions):
                image_path = os.path.join(current_dir, image_file_name)
                images.append(get_label(image_path, labels_sorted_by_first_frame, idx_by_label))
    print()
    return images, labels, idx_by_label


def default_loader(path):
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    try:
        with open(path, 'rb') as f:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(f) as img:
                return img.convert('RGB')
    except IOError:
        with open(os.path.join(error_path, date), 'a') as error_file:
            print(path)
            send_message("Error occured model probably down")
            error_file.write(path)


def get_idx_by_label(labels):
    # labels = ['preparation', 'calottriangledissection', 'cleaningcoagulation', 'gallbladderdissection', 'gallbladderretraction', 'clippingcutting', 'gallbladderpackaging']
    idx_by_label = {}

    #for i, clss in enumerate(labels):
    #    idx_by_label[clss] = i
    return idx_label
    #return idx_by_label


class Cholec80(Dataset):

    def __init__(self, root, annotations, image_file_extensions, transform=None, target_transform=None, loader=default_loader,):
        """
        @param root: root folder
        @param annotations: label folder
        @param image_file_extensions: allowed file extensions
        @param loader: sample loader
        @param transform: sample transformation
        @param target_transform: label transformation
        """
        super(Cholec80, self).__init__()
        samples, labels, idx_by_label = make_dataset(root, annotations, image_file_extensions)

        self.root = root
        self.loader = loader
        self.annotations = annotations
        self.extensions = image_file_extensions

        self.classes = labels
        self.class_to_idx = idx_by_label
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
