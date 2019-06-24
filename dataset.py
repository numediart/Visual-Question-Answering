from __future__ import print_function
import os
import json
import pickle
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from functools import lru_cache
import sys
import torchvision
import torch.nn as nn
from tqdm import tqdm
from os import listdir
from os.path import isfile, join

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


def _create_entry(question, answer):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if "test" in name:
        question_path = os.path.join(
            dataroot, 'OpenEnded_mscoco_test2015_questions.json'
        )
    else:
        if "36" in name:
            name = name[:-2]

        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' %
                      (name + '2014')
        )



    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    if "test" in name: # train, val
        entries = []
        for question in questions:
            entries.append(_create_entry(question, None))

    else:
        if "36" in name:
            name = name[:-2]
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = pickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            entries.append(_create_entry(question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data', size=224, layer="layer3", inference=False):

        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'train36', 'val36', 'val', 'test2015', 'test201536']

        #using resnet features, recommended
        if name in ['train','val','test2015']:
            features_filename = os.path.join(dataroot, "%s_resnet_%s.pkl" % (name, str(layer)))
        if name in ['train36', 'val36', 'test201536']:
            features_filename = os.path.join(dataroot, "%s.pkl" % (name))

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = pickle.load(open(ans2label_path, 'rb'))
        self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary

        self.entries = _load_dataset(dataroot, name)

        self.size = size

        resize = 256
        crop = size

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


        if inference:
            return

        print("Loading %s" % features_filename)
        if not os.path.exists(features_filename):
            print("Feature filename", features_filename,"not found, extracting...")
            self.compute_features(self.entries, dataroot, name, layer, features_filename)

        self.features = pickle.load(open(features_filename, 'rb'))

        self.v_dim = self.features[list(self.features.keys())[0]].shape[1]

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

        # if not self.use_resnet:
        #     self.features = torch.from_numpy(self.features)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']

            if None!=answer:
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

    def compute_features(self, entries, dataroot, name, layer, features_filename):
        model = torchvision.models.resnet50(pretrained=True).cuda()
        model.eval()
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        getattr(model, layer).register_forward_hook(get_activation(layer))

        base_folder = name + '2014' if 'test' != name[:4] else name

        features = {}
        for i in tqdm(range(len(entries))):
            entry = entries[i]
            img_id = entry["image_id"]
            if img_id in features:
                continue
            filename = os.path.join(dataroot, base_folder, 'COCO_%s_%s.jpg' % (str(base_folder),str(img_id).zfill(12)))
            assert os.path.exists(filename), filename + " does not exists"
            image = self._read_image(filename)
            image_np16 = image.numpy().astype(np.float16)
            inp = torch.from_numpy(image_np16).to(torch.float32)
            inp = torch.unsqueeze(inp, 0).cuda() # 1,dim,x,x
            model(inp)
            out = activation[layer].squeeze(0)
            out = out.view(out.size(0), -1)
            out = out.permute(1,0)
            features[img_id] = out.cpu().numpy().astype(np.float16)

        print("Dumping in filename %s with size %s" % (features_filename, str(len(features))))
        pickle.dump(features, open(features_filename,'wb+'))
        del model

    def __getitem__(self, index):
        entry = self.entries[index]

        img_id = entry["image_id"]
        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        question_raw = entry['question']

        target = torch.tensor(0)
        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)

        features = self.features[img_id]

        return features, question, target

    def __len__(self):
        return len(self.entries)
