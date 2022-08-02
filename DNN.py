import os
import gc
import random
import time
import pandas as pd
from tensorflow import keras
from utils import *
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.python.keras import backend as K
from sklearn.decomposition import PCA
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from numpy.random import seed
from keras.backend import clear_session

def init_layer(layer):
    session = K.get_session()
    weights_initializer = tf.variables_initializer(layer.weights)
    session.run(weights_initializer)

def get_embedding(train_d,test_d,get_scaled,n_components):
    saver = tf.train.import_meta_graph('./model/KGCN_model_concat64_n7_l2/model_1.ckpt.meta')

    from tensorflow.python import pywrap_tensorflow

    with tf.Session() as sess:
        saver.restore(sess, "./model/KGCN_model_concat64_n7_l2/model_1.ckpt")

#显示打印模型的信息
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

def get_features(data,drugid_df,use_pro):
    drug1_features = pd.merge(data,drugid_df,how='left',left_on='head',right_on='drug_id').iloc[:,4:1029].values
    drug2_features = pd.merge(data,drugid_df,how='left',left_on='tail',right_on='drug_id').iloc[:,4:1029].values
    if use_pro:
        feature = np.concatenate([drug1_features,drug2_features],axis=1)
    else:
        feature = drug1_features
    return feature

def train(i,n_components, use_pro, train_data, test_data,get_scaled):
    train_label = train_data[:, 2]
    train_label = keras.utils.to_categorical(train_label, 2)
    test_label = test_data[:, 2]
    test_label = keras.utils.to_categorical(test_label, 2)

    #print(type(test_d))
    train_feat, test_feat = get_embedding(train_data, test_data,get_scaled,n_components)
    train_data = pd.DataFrame(train_data, columns=['head', 'tail', 'label'])
    test_data = pd.DataFrame(test_data, columns=['head', 'tail', 'label'])
    clear_session()
    
    def DNN():
        train_input = Input(shape=(vector_size * 2,), name='Inputlayer')
        train_in = Dense(512, activation='relu')(train_input)
        train_in = BatchNormalization()(train_in)
        train_in = Dropout(droprate)(train_in)
        train_in = Dense(256, activation='relu')(train_in)
        train_in = BatchNormalization()(train_in)
        train_in = Dropout(droprate)(train_in)
        train_in = Dense(128, activation='relu')(train_in)
        train_in = BatchNormalization()(train_in)
        train_in = Dropout(droprate)(train_in)
        train_in = Dense(64, activation='relu')(train_in)
        train_in = BatchNormalization()(train_in)
        train_in = Dropout(droprate)(train_in)
        train_in = Dense(event_num)(train_in)
        out = Activation('softmax')(train_in)
        model = Model(input=train_input, output=out)
        # adam = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 分类交叉熵
        return model

    #DNN
    print("\n\n------------------------train: %d flod--------------------------------------------------------"%i)

    with tf.Graph().as_default():
        train_des = get_features(train_data, drugid_df, use_pro)
        test_des = get_features(test_data, drugid_df, use_pro)
        train_all_feats = np.concatenate([train_feat, train_des], axis=1)
        test_all_feats = np.concatenate([test_feat, test_des], axis=1)
        dnn = DNN()
        for layer in dnn.layers:
            init_layer(layer)

        print("start train now")

        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
        dnn.fit(train_all_feats, train_label, batch_size=1200, epochs=100, validation_data=(test_all_feats, test_label),
                callbacks=[early_stopping])
        print("--------finish training, test model!!-------------------")
        pred = dnn.predict(test_all_feats, batch_size=512)
        roc_dnn = roc_auc(test_label[:, 1], pred[:, 1])
        pr_dnn = pr_auc(test_label[:, 1], pred[:, 1])

        ##save model---------------------------------
        #savePath = "models_2500/%d_flods/"%i
        #os.makedirs(savePath,exist_ok=True)
        #dnn.save(savePath+'DNN_2500-%d.h5'%(i))
        ##----------------------------------------------

    print(roc_dnn)
    print(pr_dnn)
    p = pred[:, [1]]
    p[p >= 0.5] = 1
    p[p < 0.5] = 0

    f1, acc, Pre, Sen, Spe, MCC = scores(test_label[:, 1], p)
    train_log = {'agg': 'concat', 'dim': '64', 'AED': '12'}
    train_log['test_auc'] = roc_dnn
    train_log['test_aupr'] = pr_dnn
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_pre'] = Pre
    train_log['test_sen'] = Sen
    train_log['test_spe'] = Spe
    train_log['test_MCC'] = MCC
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log('./results/DNN_concat64_12AED_per.txt', log=train_log, mode='a')

    del dnn
    gc.collect()
    K.clear_session()
    print("-------------------finish train: %d flod--------------------------------------------------------"%i)

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
        train_log = train(i,200, True, train_data, test_data,False)
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
    write_log('./results/DNN_concat64_12AED_results.txt', temp, 'a')
    print(
        f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}, avg_pre: {temp["avg_pre"]}, avg_sen: {temp["avg_sen"]}, avg_spe: {temp["avg_spe"]}, avg_MCC: {temp["avg_MCC"]}')

seed(1)
mms = MinMaxScaler(feature_range=(0,1))
FIRST=True
event_num = 2
droprate = 0.5
vector_size = 1088

drug_id = pd.read_csv('./data/drug_index.csv',delimiter=',', header=None)
drug_id.columns = ['drug','drug_id']
drug_id = drug_id['drug_id']
#descriptors preparation
drug_feats = np.loadtxt('./data/MorganFinger.txt',delimiter=' ')
drugid_df = pd.concat([drug_id,pd.DataFrame(drug_feats)],axis=1)
DDI_softmax='./data/binary_DDI.txt'
drug1_set, drug2_set, DDI = readRecData(DDI_softmax)
cross_validation(10, np.array(DDI))







