from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from functools import lru_cache
import sys
import torchvision
import torch.nn as nn
from tqdm import tqdm

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = pickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', use_resnet=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.use_resnet = use_resnet
        self.img_id2idx = pickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), "rb"))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)


        if self.use_resnet:
            # features_filename = os.path.join(dataroot,"%s_resnet_res4f_relu.npy" % name)
            # if not os.path.exists(features_filename):
            #     print(features_filename, "not found")
            #     print("Resnet features dont exist, extracting...")
            #     self.compute_features(self.entries, dataroot, name)
            resize = 256
            crop = 224

            _transforms = []
            if resize is not None:
                _transforms.append(transforms.Resize(resize))
            if crop is not None:
                _transforms.append(transforms.CenterCrop(crop))
            _transforms.append(transforms.ToTensor())
            _transforms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]))
            self.transform = transforms.Compose(_transforms)

            self.features = []
            for e in self.entries:
                img_id = e["image_id"]
                filename = os.path.join(dataroot, '%s2014' % name, 'COCO_%s2014_%s.jpg' % (name, str(img_id).zfill(12)))
                self.features.append(filename)
            self.v_dim = 1024
            self.read_image = lru_cache(maxsize=self.__len__())(self._read_image)


        else:
            print('loading features from h5 file')
            h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
            with h5py.File(h5_path, 'r') as hf:
                self.features = np.array(hf.get('image_features'))
            self.v_dim = self.features.size(2)

        self.tokenize()
        self.tensorize()

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):

        if not self.use_resnet:
            self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None


    def _read_image(self, fname):
        with open(fname, 'rb') as f:
            img = Image.open(f).convert('RGB')
            return self.transform(img)


    def compute_features(self, entries, dataroot, name ,resize=256, crop=224):
        _transforms = []
        if resize is not None:
            _transforms.append(transforms.Resize(resize))
        if crop is not None:
            _transforms.append(transforms.CenterCrop(crop))
        _transforms.append(transforms.ToTensor())
        _transforms.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(_transforms)

        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        res4f_relu = nn.Sequential(*list(resnet.children())[:-3]).cuda()
        for param in res4f_relu.parameters():
            param.requires_grad = False
        features = []
        for i in tqdm(range(len(entries))):
            entry = entries[i]
            img_id = entry["image_id"]
            filename = os.path.join(dataroot, '%s2014' % name, 'COCO_%s2014_%s.jpg' % (name, str(img_id).zfill(12)))
            assert os.path.exists(filename), filename+" does not exists"
            image = self._read_image(filename)
            features.append(res4f_relu(torch.unsqueeze(image, 0).cuda()))

        x = torch.stack(features).numpy()
        np.save(os.path.join(dataroot,"%s_resnet_res4f_relu.npy" % name), x.astype(np.float16))



    def __getitem__(self, index):
        entry = self.entries[index]
        if self.use_resnet:
            features = self.read_image(self.features[index])
        else:
            features = self.features[entry['image']]

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, question, target

    def __len__(self):
        return len(self.entries)
