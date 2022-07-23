import os
import json
import gc
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from abc import abstractmethod
from tensorflow.keras import backend as K
from ampligraph.utils import save_model,restore_model
from sklearn.model_selection import train_test_split, StratifiedKFold

def readTriple(path,sep=None):
    with open(path,'r',encoding='utf-8') as f:
        for line in f.readlines():
            if sep:
                lines = line.strip().split()
            else:
                lines=line.strip().split()
            yield lines

def construct_kg(kgTriples):
    print('Generate knowledge graph index')
    kg = dict()
    for triple in kgTriples:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))
    return kg

def readRecData(path,test_ratio=0.2):
    print('Reading DDI triplets...')
    drug1_set,drug2_set=set(),set()
    DDIlabel=[]
    for d1,  d2, label in tqdm(readTriple(path)):
        drug1_set.add(int(d1))
        drug2_set.add(int(d2))
        DDIlabel.append((int(d1),int(d2),int(label)))
    return list(drug1_set),list(drug2_set),DDIlabel

def readKgData(path):
    print('Reading KG triplets...')
    entity_set, relation_set = set(), set()
    triples = []
    for h, r, t in tqdm(readTriple(path)):
        entity_set.add(int(h))
        entity_set.add(int(t))
        relation_set.add(int(r))
        triples.append([int(h), int(r), int(t)])

    # 返回实体集合列表，关系集合列表，与三元组列表
    return list(entity_set), list(relation_set), triples

def construct_adj(neighbor_sample_size, kg, entity_num):
    print('Generate entity adjacency lists and relation adjacency lists')
    adj_entity = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    adj_relation = np.zeros([entity_num, neighbor_sample_size], dtype=np.int64)
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=neighbor_sample_size, replace=True)
        adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])
    return adj_entity, adj_relation

#KGCN model
LAYER_IDS = {}

def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]

class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations, drug1_embeddings):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations, drug1_embeddings)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, drug1_embeddings):
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, drug1_embeddings):
        avg = False
        if not avg:
            drug1_embeddings = tf.reshape(drug1_embeddings, [self.batch_size, 1, 1, self.dim])
            drug1_relation_scores = tf.reduce_mean(drug1_embeddings * neighbor_relations, axis=-1)
            drug1_relation_scores_normalized = tf.nn.softmax(drug1_relation_scores, dim=-1)
            drug1_relation_scores_normalized = tf.expand_dims(drug1_relation_scores_normalized, axis=-1)
            neighbors_aggregated = tf.reduce_mean(drug1_relation_scores_normalized * neighbor_vectors, axis=2)
        else:
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated

class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, drug1_embeddings):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, drug1_embeddings)
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])
        return self.act(output)

class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, drug1_embeddings):
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, drug1_embeddings)
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, drug1_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, drug1_embeddings)
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)

import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m

class KGCN(object):
    def __init__(self, args, n_drug1, n_entity, n_relation, adj_entity, adj_relation):
        self._parse_args(args, adj_entity, adj_relation)
        self._build_inputs()
        self._build_model(n_drug1, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        print("_parse_args" + str(adj_entity.shape))

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.drug1_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='drug1_indices')
        self.drug2_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='drug2_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')  # 具体的label(0/1)

    def _build_model(self, n_drug1, n_entity, n_relation):
        self.drug1_emb_matrix = tf.get_variable(
            shape=[n_drug1, self.dim], initializer=KGCN.get_initializer(),
            name='drug1_emb_matrix')  # get_initializer()初始化变量
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        self.drug1_embeddings = tf.nn.embedding_lookup(self.drug1_emb_matrix,
                                                      self.drug1_indices)

        entities, relations = self.get_neighbors(self.drug2_indices)

        self.drug2_embeddings, self.aggregators = self.aggregate(entities, relations)
        self.scores = tf.reduce_sum(self.drug1_embeddings * self.drug2_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        print("get_neighbors")
        print(seeds.shape)
        seeds = tf.expand_dims(seeds, axis=1)
        print(seeds.shape)
        entities = [seeds]
        print(len(entities))
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def aggregate(self, entities, relations):
        print("aggregate")

        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []

            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                    drug1_embeddings=self.drug1_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.drug1_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)

        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)

        acc = accuracy_score(y_true=labels, y_pred=scores)
        precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=scores)
        aupr = m.auc(recall, precision)

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for j in range(len(labels)):
            if labels[j] == 1:
                if labels[j] == scores[j]:
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                if labels[j] == scores[j]:
                    tn = tn + 1
                else:
                    fp = fp + 1
        if tp == 0 and fp == 0:
            sensitivity = float(tp) / (tp + fn)
            specificity = float(tn) / (tn + fp)
            precision = 0
            MCC = 0
        else:
            sensitivity = float(tp) / (tp + fn)
            specificity = float(tn) / (tn + fp)
            precision = float(tp) / (tp + fp)
            MCC = float(tp * tn - fp * fn) / (np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))

        Pre = np.float64(precision)
        Sen = np.float64(sensitivity)
        Spe = np.float64(specificity)
        MCC = np.float64(MCC)
        return auc, f1, acc, aupr, Pre, Sen, Spe, MCC

    def get_scores(self, sess, feed_dict):
        return sess.run([self.drug2_indices, self.scores_normalized], feed_dict)



def get_feed_dict(model, data, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    return feed_dict


def ctr_eval(sess, model, data, batch_size):
    start = 0
    auc_list = []
    f1_list = []

    acc_list = []
    aupr_list = []
    Pre_list = []
    Sen_list = []
    Spe_list = []
    MCC_list = []

    while start + batch_size <= data.shape[0]:
        auc, f1, acc, aupr, Pre, Sen, Spe, MCC = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)

        acc_list.append(acc)
        aupr_list.append(aupr)
        Pre_list.append(Pre)
        Sen_list.append(Sen)
        Spe_list.append(Spe)
        MCC_list.append(MCC)

        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(acc_list)), float(
        np.mean(aupr_list)), float(np.mean(Pre_list)), float(np.mean(Sen_list)), float(np.mean(Spe_list)), float(
        np.mean(MCC_list))


def write_log(filename: str, log, mode='w'):
    with open(filename, mode) as writers:
        writers.write('\n')
        json.dump(log, writers, indent=4, ensure_ascii=False)


def train(args, n_user, n_item, n_entity, n_relation, train_data, eval_data, test_data, adj_entity, adj_relation,
          show_loss):
    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    saver = tf.compat.v1.train.Saver()

    # top-K evaluation settings
    # user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size
            if show_loss:
                print(start, loss)

            # CTR evaluation
            train_auc, train_f1, train_acc, train_aupr, train_Pre, train_Sen, train_Spe, train_MCC = ctr_eval(sess,
                                                                                                              model,
                                                                                                              train_data,
                                                                                                              args.batch_size)
            eval_auc, eval_f1, eval_acc, eval_aupr, eval_Pre, eval_Sen, eval_Spe, eval_MCC = ctr_eval(sess, model,
                                                                                                      eval_data,
                                                                                                      args.batch_size)
            test_auc, test_f1, test_acc, test_aupr, test_Pre, test_Sen, test_Spe, test_MCC = ctr_eval(sess, model,
                                                                                                      test_data,
                                                                                                      args.batch_size)

            print(
                'epoch %d    train_auc: %.4f  train_f1: %.4f    train_acc: %.4f  train_aupr: %.4f    train_pre: %.4f  train_sen: %.4f train_spe: %.4f train_MCC: %.4f '
                % (step, train_auc, train_f1, train_acc, train_aupr, train_Pre, train_Sen, train_Spe, train_MCC))

            print(
                'epoch %d    eval_auc: %.4f  eval_f1: %.4f    eval_acc: %.4f  eval_aupr: %.4f    eval_pre: %.4f  eval_sen: %.4f eval_spe: %.4f eval_MCC: %.4f'
                % (step, eval_auc, eval_f1, eval_acc, eval_aupr, eval_Pre, eval_Sen, eval_Spe, eval_MCC))

            print(
                'epoch %d    test_auc: %.4f  test_f1: %.4f    test_acc: %.4f  test_aupr: %.4f    test_pre: %.4f  test_sen: %.4f test_spe: %.4f test_MCC: %.4f'
                % (step, test_auc, test_f1, test_acc, test_aupr, test_Pre, test_Sen, test_Spe, test_MCC))

            train_log = {'epoch': step}
            train_log['dev_auc'] = eval_auc
            train_log['dev_acc'] = eval_acc
            train_log['dev_f1'] = eval_f1
            train_log['dev_aupr'] = eval_aupr
            train_log['dev_pre'] = eval_Pre
            train_log['dev_sen'] = eval_Sen
            train_log['dev_spe'] = eval_Spe
            train_log['dev_MCC'] = eval_MCC
            train_log['test_auc'] = test_auc
            train_log['test_acc'] = test_acc
            train_log['test_f1'] = test_f1
            train_log['test_aupr'] = test_aupr
            train_log['test_pre'] = test_Pre
            train_log['test_sen'] = test_Sen
            train_log['test_spe'] = test_Spe
            train_log['test_MCC'] = test_MCC
            train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            write_log('C:/Users/zou/Desktop/zj_NFM/kgcn_per_concat64_n7_l2.txt', log=train_log, mode='a')

        save_path = saver.save(sess, "KGCN_model_concat64_n7_l2/model_1.ckpt")

    del model
    gc.collect()
    K.clear_session()
    return train_log

def cross_validation(K_fold, examples):
    subsets = dict()
    n_subsets = int(len(examples) / K_fold)
    remain = set(range(0, len(examples) - 1))
    for i in reversed(range(0, K_fold - 1)):
        subsets[i] = random.sample(remain, n_subsets)
        remain = remain.difference(subsets[i])
    subsets[K_fold - 1] = remain

    temp = {'avg_auc': 0.0, 'avg_acc': 0.0, 'avg_f1': 0.0, 'avg_aupr': 0.0, 'avg_pre': 0.0, 'avg_sen': 0.0,
            'avg_spe': 0.0, 'avg_MCC': 0.0}

    for i in reversed(range(0, K_fold)):
        test_d = examples[list(subsets[i])]
        eval_data, test_data = train_test_split(test_d, test_size=0.5)
        train_d = []

        for j in range(0, K_fold):
            if i != j:
                train_d.extend(examples[list(subsets[j])])
        train_data = np.array(train_d)

        train_log = train(args, n_drug1, n_drug2, n_entity, 8, train_data, eval_data, test_data, adj_entity, adj_relation,
                          False)

        temp['avg_auc'] = temp['avg_auc'] + train_log['test_auc']
        temp['avg_acc'] = temp['avg_acc'] + train_log['test_acc']
        temp['avg_f1'] = temp['avg_f1'] + train_log['test_f1']
        temp['avg_aupr'] = temp['avg_aupr'] + train_log['test_aupr']
        temp['avg_pre'] = temp['avg_pre'] + train_log['test_pre']
        temp['avg_sen'] = temp['avg_sen'] + train_log['test_sen']
        temp['avg_spe'] = temp['avg_spe'] + train_log['test_spe']
        temp['avg_MCC'] = temp['avg_MCC'] + train_log['test_MCC']
        temp['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    for key in temp:
        if key == 'timestamp':
            continue
        temp[key] = temp[key] / K_fold
    write_log('C:/Users/zou/Desktop/zj_NFM/kgcn_concat_dim64_n7_l2_results.txt', temp, 'a')
    print(
        f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, '
        f'avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}, avg_pre: {temp["avg_pre"]}, avg_sen: {temp["avg_sen"]}, '
        f'avg_spe: {temp["avg_spe"]}, avg_MCC: {temp["avg_MCC"]}')


kg_index = './data/kg_index.txt'
DDI_index = './data/DDI_index.txt'
drug1_set, drug2_set ,DDIlabel = readRecData(DDI_index)
n_drug1 = len(drug1_set)
n_drug2 = len(drug2_set)
n_entity = len(DDIlabel)
entitys, relations, kgTriples = readKgData(kg_index)
adj_kg = construct_kg(kgTriples)
adj_entity, adj_relation = construct_adj(7, adj_kg, len(entitys))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--aggregator', type=str, default='concat', help='which aggregator to use')
parser.add_argument('--dim', type=int, default=64, help='dimension of user and entity embeddings')
parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity representation')
parser.add_argument('--l2_weight', type=float, default=1e-5, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
parser.add_argument('--neighbor_sample_size', type=int, default=7, help='the number of neighbors to be sampled')
args = parser.parse_args(args=[])
np.random.seed(555)
cross_validation(10,np.array(DDIlabel))


