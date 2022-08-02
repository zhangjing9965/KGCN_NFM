import os
import gc
import time
import random
import pandas as pd
import tensorflow as tf
from utils import *
from tensorflow import keras
from deepctr.models import NFM
from deepctr.feature_column import SparseFeat,DenseFeat,get_feature_names
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam,Adagrad,Adamax
from tensorflow.python.keras import backend as K

def get_nfm_input(re_train_all,re_test_all,train_feat,test_feat,train_des,test_des,embedding_dim,pca_components):
    train_all_feats = np.concatenate([train_feat,train_des],axis=1)
    test_all_feats = np.concatenate([test_feat,test_des],axis=1)
    train_all_feats = mms.fit_transform(train_all_feats)
    test_all_feats = mms.transform(test_all_feats)
    feature_columns = [SparseFeat('head',re_train_all['head'].unique().shape[0],embedding_dim=embedding_dim),
                        SparseFeat('tail',re_train_all['tail'].unique().shape[0],embedding_dim=embedding_dim),
                        DenseFeat("feats",train_all_feats.shape[1]),
                        #DenseFeat("des",train_des.shape[1])
                        ]
    train_model_input = {'head':LE.transform(re_train_all['head'].values),
                    'tail':LE.transform(re_train_all['tail'].values),
                     'feats':train_all_feats,
                     #'des':train_des
                    }
    test_model_input = {'head':LE.transform(re_test_all['head'].values),
                    'tail':LE.transform(re_test_all['tail'].values),
                    'feats':test_all_feats,
                     #'des':test_des
                    }
    return feature_columns,train_model_input,test_model_input

def get_features(data,drugid_df,use_pro):
    drug1_features = pd.merge(data,drugid_df,how='left',left_on='head',right_on='drug_id').iloc[:,4:1029].values
    drug2_features = pd.merge(data,drugid_df,how='left',left_on='tail',right_on='drug_id').iloc[:,4:1029].values
    if use_pro:
        feature = np.concatenate([drug1_features,drug2_features],axis=1)
    else:
        feature = drug1_features
    return feature

def train_nfm(feature_columns, train_model_input, train_label, test_model_input, y, patience):
    re_model = NFM(feature_columns, feature_columns, task='binary', dnn_hidden_units=(128, 128),
                   l2_reg_dnn=1e-5, l2_reg_linear=1e-5,
                   )
    re_model.compile(Adam(1e-3), "binary_crossentropy",
                     metrics=[keras.metrics.Precision(name='precision'), ], )
    es = EarlyStopping(monitor='loss', patience=patience, min_delta=0.0001, mode='min', restore_best_weights=True)
    history = re_model.fit(train_model_input, train_label,
                           batch_size=20000, epochs=100,
                           verbose=2,
                           callbacks=[es]
                           )
    #re_model.save_weights('re_model_weight1.h5')
    pred_y = re_model.predict(test_model_input, batch_size=512)
    roc_nfm = roc_auc(y, pred_y[:, 0])
    pr_nfm = pr_auc(y, pred_y[:, 0])
    print(roc_nfm)
    print(pr_nfm)
    pred_y[pred_y >= 0.5] = 1
    pred_y[pred_y < 0.5] = 0

    f1, acc, Pre, Sen, Spe, MCC = scores(y, pred_y)
    train_log = {'agg':'concat','dim':'64','AED':'12AED'}
    train_log['test_auc'] = roc_nfm
    train_log['test_aupr'] = pr_nfm
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_pre'] = Pre
    train_log['test_sen'] = Sen
    train_log['test_spe'] = Spe
    train_log['test_MCC'] = MCC
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log('./results/KGCN_NFM_per_lr_005_concat64_n7_L2_Morgan.txt', log=train_log, mode='a')

    del re_model
    gc.collect()  # 清理内存
    K.clear_session()

    return train_log

def get_embedding(train_d,test_d,get_scaled,n_components):
    saver = tf.train.import_meta_graph('./model/KGCN_model_concat64_n7_l2/model_1.ckpt.meta')
    from tensorflow.python import pywrap_tensorflow

    with tf.Session() as sess:
        saver.restore(sess, "./model/KGCN_model_concat64_n7_l2/model_1.ckpt")

    model_dir = "./"
    checkpoint_path = os.path.join(model_dir,"./model/KGCN_model_concat64_n7_l2/model_1.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        #print("tensor_name: ", key)
        if key=='entity_emb_matrix/Adam_1':
            entity_emb_matrix=reader.get_tensor(key)
    #print(entity_emb_matrix.shape)
    #print(test_d[:, 0])



    train_drug1 = tf.nn.embedding_lookup(entity_emb_matrix, train_d[:, 0])
    train_drug2 = tf.nn.embedding_lookup(entity_emb_matrix, train_d[:, 1])
    test_drug1 = tf.nn.embedding_lookup(entity_emb_matrix, test_d[:, 0])
    test_drug2 = tf.nn.embedding_lookup(entity_emb_matrix, test_d[:, 1])
    with tf.Session() as sess:
        train_drug1=train_drug1.eval()
        train_drug2=train_drug2.eval()
        test_drug1=test_drug1.eval()
        test_drug2=test_drug2.eval()
    train_feat=np.concatenate([train_drug1,train_drug2],axis=1)
    test_feat=np.concatenate([test_drug1,test_drug2],axis=1)
    train_feat=mms.fit_transform(train_feat)
    test_feat=mms.fit_transform(test_feat)
    if get_scaled:
        pca = PCA(n_components=n_components)
        train_feat = pca.fit_transform(train_feat)  #降维
        test_feat = pca.transform(test_feat)  #归一化
    else:
        train_feat = train_feat
        test_feat = test_feat

    return train_feat,test_feat

def train(embedding_dim, n_components, use_pro, patience, train_data, test_data,get_scaled):
    train_feat, test_feat = get_embedding(train_data, test_data,get_scaled,n_components)
    # nfm preparation
    train_data = pd.DataFrame(train_data, columns=['head', 'tail', 'label'])
    test_data = pd.DataFrame(test_data, columns=['head', 'tail', 'label'])
    train_label = train_data['label']
    test_label = test_data['label']

    train_des = get_features(train_data, drugid_df, use_pro)
    test_des = get_features(test_data, drugid_df, use_pro)
    feature_columns, train_model_input, test_model_input = get_nfm_input(train_data, test_data,
                                                                         train_feat, test_feat,
                                                                         train_des, test_des,
                                                                         embedding_dim, n_components)
    train_log = train_nfm(feature_columns, train_model_input, train_label, test_model_input, test_label, patience)
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
        test_data = examples[list(subsets[i])]
        train_data = []
        for j in range(0, K_fold):

            if i != j:
                train_data.extend(examples[list(subsets[j])])
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        print(i)
        train_log = train(100, 200, True, 5, train_data, test_data,False)

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
    write_log('./results/KGCN_NFM_concat64_n7_lr005_l2_results_Morgan.txt', temp, 'a')
    print(
        f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}, avg_pre: {temp["avg_pre"]}, avg_sen: {temp["avg_sen"]}, avg_spe: {temp["avg_spe"]}, avg_MCC: {temp["avg_MCC"]}')

DDI_index='./data/Samples_12AED1.csv'
drug1_set, drug2_set ,DDI = readRecData(DDI_index)
drug_id = pd.read_csv('./data/drug_index.csv',delimiter=',', header=None)
drug_id.columns = ['drug','drug_id']
drug_id = drug_id['drug_id']
#descriptors preparation
drug_feats = np.loadtxt('./data/MorganFinger.txt',delimiter=' ')
drugid_df = pd.concat([drug_id,pd.DataFrame(drug_feats)],axis=1)
LE = LabelEncoder()
LE.fit(drugid_df['drug_id'].values)
mms = MinMaxScaler(feature_range=(0,1))
cross_validation(10, np.array(DDI))

