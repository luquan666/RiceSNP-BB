import csv
import pandas as pd
import numpy as np
import keras
from amber.architect.modelSpace import Operation, ModelSpace
from amber.modeler import KerasResidualCnnBuilder
from keras import Model
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

model_space = ModelSpace()

model_space.add_layer(0,[
            Operation('conv1d', filters=8, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=8, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=8, pool_size=3, strides=1),
            Operation('avgpool1d', filters=8, pool_size=3, strides=1),
            Operation('identity', filters=8),
      ])
model_space.add_layer(1,[
            Operation('conv1d', filters=8, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=8, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=8, pool_size=3, strides=1),
            Operation('avgpool1d', filters=8, pool_size=3, strides=1),
            Operation('identity', filters=8),
      ])
model_space.add_layer(2,[
            Operation('conv1d', filters=16, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=16, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=16, pool_size=3, strides=1),
            Operation('avgpool1d', filters=16, pool_size=3, strides=1),
            Operation('identity', filters=16),
      ])

model_space.add_layer(3,[
            Operation('conv1d', filters=16, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=16, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=16, pool_size=3, strides=1),
            Operation('avgpool1d', filters=16, pool_size=3, strides=1),
            Operation('identity', filters=16),
      ])

model_space.add_layer(4,[
            Operation('conv1d', filters=32, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=32, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=32, pool_size=3, strides=1),
            Operation('avgpool1d', filters=32, pool_size=3, strides=1),
            Operation('identity', filters=32),
      ])


model_space.add_layer(5,[
            Operation('conv1d', filters=32, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=32, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=32, pool_size=3, strides=1),
            Operation('avgpool1d', filters=32, pool_size=3, strides=1),
            Operation('identity', filters=32),
      ])

data_train=pd.read_csv('Rearrange_101nt_features.csv')
X=np.array(data_train)
X_train = X.reshape(10472,25,4)
y = [1]*5236+[0]*5236
Y_train=np.array(y)
data_train1=pd.read_csv('independent_dataset_features.csv',header=None)
X_test=np.array(data_train1)
X_test=X_test.reshape(244,25,4)
y_test = [1]*122+[0]*122
y_test=np.array(y_test)
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

def read_train_data():
    return x_train,y_train
def read_val_data():
    return x_val,y_val
def read_val_data():
    return X_test,y_test

val_data = read_val_data()


train_data= read_train_data()
input_node = Operation('input', shape=(25,4), name="input")
output_node = Operation('dense', units=1, activation='sigmoid')
keras_builder = KerasResidualCnnBuilder(
    inputs_op=input_node,
    output_op=output_node,
    fc_units=100,
    flatten_mode='Flatten',
    model_compile_dict={
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    },
    model_space=model_space,
    dropout_rate=0.1,
    wsf=2
)

csv_file='test.csv'
best_arc = [4,3,0,1,0,1,1,0,1,1,0,1,1,1,0,3,1,1,0,0,0]
searched_mod = keras_builder(best_arc)
searched_mod.optimizer.lr = 0.0001
searched_mod.summary()
searched_mod.fit(
        X_train,
        Y_train,
        batch_size=32,
        validation_data=val_data,
        epochs=50,
        verbose=1,
        )
y_pred_val = searched_mod.predict(X_test)
result_val_fold = pd.DataFrame({'y_test': y_test.flatten(), 'y_pred_test': y_pred_val.flatten()})
result_val_fold.to_csv(f'test_results.csv', index=False)
def calculate_val(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len( y_pred)):
        if (y_pred[i] >= 0.5) & (y_true[i] == 1):
            TP += 1
        if (y_pred[i] < 0.5) & (y_true[i] == 0):
            TN += 1
        if (y_pred[i] >= 0.5) & (y_true[i] == 0):
            FP += 1
        if (y_pred[i] < 0.5) & (y_true[i] == 1):
            FN += 1
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * TP / (2 * TP + FP + FN)
    mcc = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + np.finfo(float).eps)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    aupr = auc(recall, precision)
    precision = TP / (TP + FP)
    auc_score = roc_auc_score(y_true, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    performance_metrics = {
        'val'
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1_score,
        'MCC': mcc,
        'AUPR': aupr,
        'AUC': auc_score,
        'Precision': precision,
        'Accuracy': accuracy
    }

    return performance_metrics
performance = calculate_val(y_test, y_pred_val)
with open(csv_file, mode='a') as file:
    writer = csv.DictWriter(file, fieldnames=performance.keys())
    if file.tell() == 0:
        writer.writeheader()
    writer.writerow(performance)
print(performance)






