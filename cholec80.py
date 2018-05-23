from torch.utils.data.dataset import Dataset

from PIL import Image

import re
import os


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(video_dir, annotations_dir):
    """Finds all labels from a given video path and a annotations path"""
    video_number = video_dir[-2:]
    annotations_file = 'video{}-phase.txt'.format(video_number)
    classes = []
    thresholds = []
    class_threshold = {}
    with open(os.path.join(annotations_dir,annotations_file), 'r') as file:
        for clss in file:
            clss = clss.lower()
            interval = re.sub("\D", "", clss)
            formatted_clss = re.sub("\d+\s+", "", clss).strip()
            if formatted_clss not in classes:
                classes.append(formatted_clss)
                thresholds.append(interval)
    # Remove the frame, phase declaration from the list and the dict
    classes.pop(0)
    thresholds.pop(0)
    for label, threshold in zip(classes, thresholds):
        class_threshold[int(threshold)] = label
    return classes, class_threshold


def get_label(path, labels):
    """Returns the label for a single image"""
    frame_name = re.sub(".*\d+\/", "", path)
    frame_number = int(re.sub("\D+$", "", frame_name)) * 25
    sorted_labels = sorted(labels.items())
    clss = 'preparation'
    for threshold, label in sorted_labels:
        if frame_number < int(threshold):
            return path, clss
        clss = label
    return path, sorted_labels[-1][1]


def make_dataset(dir, annotations_dir, extensions):
    images = []
    for root, dir_names, file_names in sorted(os.walk(dir)):
        if dir_names:
            continue
        classes, class_threshold = find_classes(root, annotations_dir)
        for filename in file_names:
            if has_file_allowed_extension(filename, extensions):
                path = os.path.join(root, filename)
                item = get_label(path, class_threshold)
                images.append(item)
    return images


def default_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Cholec80(Dataset):

    def __init__(self, root, annotations, extensions, transform=None, target_transform=None, loader=default_loader,):
        """
        @param root: root folder
        @param annotations: label folder
        @param extensions: allowed file extensions
        @param loader: sample loader
        @param transform: sample transformation
        @param target_transform: label transformation
        """
        super(Cholec80, self).__init__()
        samples = make_dataset(root, annotations, extensions)
        self.root = root
        self.loader = loader
        self.annotations = annotations
        self.extensions = extensions
        self.transform = transform
        self.target_transform = target_transform
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        print('path: ', path)
        print('label: ', target)
        sample = self.loader(path)
        print(sample)
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
