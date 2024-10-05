import csv

try:
  import amber
  print('AMBER imported successfully')
except ModuleNotFoundError as e:
  print('You need to restart your colab runtime for AMBER to take effect.')
  print('Go to Runtime-->Restart Runtime and run all')
  raise e
import os
import shutil

try:
  from amber import Amber
  from amber.architect import ModelSpace, Operation
except ModuleNotFoundError:
  print('Restart your Colab runtime by Runtime->restart runtime, and run this cell again')


def get_model_space(out_filters=32, num_layers=9, num_pool=3):
    model_space = ModelSpace()
    expand_layers = [num_layers//num_pool*i-1 for i in range(num_pool)]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            Operation('maxpool1d', filters=out_filters, pool_size=3, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=3, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space
import pandas as pd
import numpy as np
data_train=pd.read_csv('Rearrange_101nt_features.csv')
X=np.array(data_train)
X_train = X.reshape(10472,25,4)
y = [1]*5236+[0]*5236
Y_train=np.array(y)
idx = np.random.permutation(len(Y_train))
X_train= X_train[idx]
Y_train= Y_train[idx]
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
def read_train_data():
    return x_train,y_train

def read_val_data():
    return x_val,y_val

val_data = read_val_data()
train_data= read_train_data()

type_dict = {
    'controller_type': 'GeneralController',
    'modeler_type': 'EnasCnnModelBuilder',
    'knowledge_fn_type': 'zero',
    'reward_fn_type': 'LossAucReward',
    'manager_type': 'EnasManager',
    'env_type': 'EnasTrainEnv'
}


wd = "./outputs/AmberDeepSea/"
if os.path.isdir(wd):
    shutil.rmtree(wd)
os.makedirs(wd)
input_node = Operation('input', shape=(25,4), name="input")
output_node = Operation('dense', units=1, activation='sigmoid')
model_compile_dict = {
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
}

model_space = get_model_space(out_filters=8, num_layers=6)
print(model_space)
specs = {
    'model_space': model_space,
    'controller': {
        'share_embedding': {i: 0 for i in range(1, len(model_space))},
        'with_skip_connection': True,
        'skip_weight': 1.0,
        'skip_target': 0.4,
        'kl_threshold': 0.01,
        'train_pi_iter': 10,
        'buffer_size': 1,
        'batch_size': 20
    },

    'model_builder': {
        'dag_func': 'EnasConv1dDAG',
        'batch_size': 128,
        'inputs_op': [input_node],
        'outputs_op': [output_node],
        'model_compile_dict': model_compile_dict,
        'dag_kwargs': {
            'stem_config': {
                'flatten_op': 'flatten',
                'fc_units': 100,
            }
        }
    },

    'knowledge_fn': {'data': None, 'params': {}},

    'reward_fn': {'method': 'auc'},

    'manager': {
        'data': {
            'train_data': train_data,
            'validation_data':val_data,
            # 'test_data':test_data
        },
        'params': {
            'epochs': 1,
            'child_batchsize': 128,
            'store_fn': 'minimal',
            'working_dir': wd,
            'verbose': 2
        }
    },

    'train_env': {
        'max_episode': 10,
        'max_step_per_ep': 10,
        'working_dir': wd,
        'time_budget': "00:15:00",
        'with_input_blocks': False,
        'with_skip_connection': True,
        'child_train_steps': 10,
    }
}

amb = Amber(types=type_dict, specs=specs)
amb.controller
amb.run()

from amber.utils.io import read_history
hist = read_history([os.path.join(wd, "train_history.csv")],
                    metric_name_dict={'zero':0, 'auc': 1})
hist = hist.sort_values(by='auc', ascending=False)
hist.head(n=5)
from amber.modeler import KerasResidualCnnBuilder

print(model_space)
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

best_arc = hist.iloc[0][[x for x in hist.columns if x.startswith('L')]].tolist()
searched_mod=keras_builder(best_arc)
searched_mod.summary()

from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
num_folds = 5
csv_file = 'val_results.csv'
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)  # 只需要一次5折交叉验证
for fold, (train_indexes, val_indexes) in enumerate(skf.split(X_train, Y_train)):
    x_train, y_train = X_train[train_indexes], Y_train[train_indexes]
    X_val, y_val = X_train[val_indexes], Y_train[val_indexes]
    val_data = (X_val, y_val)
    model = keras_builder(best_arc)
    model.optimizer.lr = 0.0001
    model.fit(
        x_train,
        y_train,
        batch_size=32,
        validation_data=val_data,
        epochs=50,
        verbose=1,
    )
    y_pred_val = model.predict(X_val).flatten()
    result_val_fold = pd.DataFrame({'y_test': y_val.flatten(), 'y_pred_test': y_pred_val})
    result_val_fold.to_csv(f'val_results_fold{fold}.csv', index=False)
    def calculate_val(y_true, y_pred):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(X_val)):
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
            'Fold': fold,
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
    performance = calculate_val(y_val, y_pred_val)
    with open(csv_file, mode='a') as file:
        writer = csv.DictWriter(file, fieldnames=performance.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(performance)
    print(performance)



