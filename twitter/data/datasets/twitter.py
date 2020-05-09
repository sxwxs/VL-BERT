import os
import time
import jsonlines
import json
import _pickle as cPickle
from PIL import Image
from copy import deepcopy

import torch
from torch.utils.data import Dataset
from external.pytorch_pretrained_bert import BertTokenizer, BasicTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist
from common.utils.mask import generate_instance_mask
from common.nlp.misc import get_align_matrix
from common.utils.misc import block_digonal_matrix
from common.nlp.misc import random_word_with_token_ids
from common.nlp.roberta import RobertaTokenizer


DATA_path =  '/home/sxw/jupyter_workspace/Data/'
base_path = DATA_path+'sarcasm/'
text_path = base_path+'text/'
imag_path = base_path+'dataset_image/'

class TwitterDataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, transform=None, task='Q2A', test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 basic_tokenizer=None, tokenizer=None, pretrained_model_name=None,
                 only_use_relevant_dets=False, add_image_as_a_box=False, mask_size=(14, 14),
                 aspect_grouping=False, basic_align=False, qa2r_noq=False, qa2r_aug=False,
                 seq_len=64,
                 **kwargs):
        """
        Visual Commonsense Reasoning Dataset

        :param ann_file: annotation jsonl file
        :param image_set: image folder name, e.g., 'vcr1images'
        :param root_path: root path to cache database loaded from annotation file
        :param data_path: path to vcr dataset
        :param transform: transform
        :param task: 'Q2A' means question to answer, 'QA2R' means question and answer to rationale,
                     'Q2AR' means question to answer and rationale
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param only_use_relevant_dets: filter out detections not used in query and response
        :param add_image_as_a_box: add whole image as a box
        :param mask_size: size of instance mask of each object
        :param aspect_grouping: whether to group images via their aspect
        :param basic_align: align to tokens retokenized by basic_tokenizer
        :param qa2r_noq: in QA->R, the query contains only the correct answer, without question
        :param qa2r_aug: in QA->R, whether to augment choices to include those with wrong answer in query
        :param kwargs:
        """
        super(TwitterDataset, self).__init__()
        self.cache_dir = os.path.join(root_path, 'cache')
        assert not cache_mode, 'currently not support cache mode!'
        
        self.data_path = data_path
        self.test_mode = test_mode
        self.ann_file = os.path.join(text_path, ann_file)
        self.image_set = image_set
        self.transform = transform
        self.cache_mode = cache_mode
        self.basic_tokenizer = basic_tokenizer if basic_tokenizer is not None \
            else BasicTokenizer(do_lower_case=True)
        if tokenizer is None:
            if pretrained_model_name is None:
                pretrained_model_name = 'bert-base-uncased'
            if 'roberta' in pretrained_model_name:
                tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name, cache_dir=self.cache_dir)
            else:
                tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, cache_dir=self.cache_dir)
        self.tokenizer = tokenizer
        self.database = self.load_annotations(self.ann_file)

    @property
    def data_names(self):
        if self.test_mode:
            return ['image', 'boxes', 'im_info', 'question']
        else:
            return ['image', 'boxes', 'im_info', 'question', 'label']

    def load_annotations(self, ann_file):
        tic = time.time()
        database = []
        file = open(ann_file)
        for line in file:
            lineLS = eval(line)
            image_id = lineLS[0]
            image_path = ((imag_path + '%s.jpg') % image_id)
            sentence = lineLS[1]
            label = lineLS[-1]
            if not os.path.exists('/tmp/sarcasm_image2/%s.boxs' % image_id):
                continue
            db_i = {
                'annot_id': image_id,
                'img_fn': image_path,
                'box_fn': '/tmp/sarcasm_image2/%s.boxs' % image_id,
                'text': sentence,
                'label': label,
            }
            database.append(db_i)
        file.close()
        print('Done (t={:.2f}s)'.format(time.time() - tic))
        return database

    def __getitem__(self, index):
        # self.person_name_id = 0
        idb = deepcopy(self.database[index])
        with open(idb['box_fn']) as f:
            idb['boxes'] = json.load(f)
        # idb['width'] = metadata['width']
        # idb['height'] = metadata['height']

        objects_replace_name = []
        
        idb['text'] = self.tokenizer.convert_tokens_to_ids(idb['text'])


        image = self._load_image(idb['img_fn'])
        w0, h0 = image.size

        # extract bounding boxes and instance masks in metadata
        # = torch.zeros((len(idb['boxes']), 6))
        #if len(idb['boxes']) > 0:
        boxes = torch.tensor(idb['boxes'])
        
        im_info = torch.tensor([w0, h0, 1.0, 1.0])
        flipped = False
        if self.transform is not None:
            image, boxes, _, im_info, flipped = self.transform(image, boxes, None, im_info, flipped)
        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)
        if self.test_mode:
            return image, boxes, im_info, idb['text']
        else:
            # print([(self.answer_vocab[i], p.item()) for i, p in enumerate(label) if p.item() != 0])
            return image, boxes, im_info, idb['text'], idb['label']

    def __len__(self):
        return len(self.database)

    def _load_image(self, path):
        return Image.open(path)
