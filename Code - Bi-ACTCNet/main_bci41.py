import time
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
import cnn_model
from preprocess_bci41 import get_data
import os

def train(dataset_conf, train_conf, results_path):
    in_exp = time.time()

    best_models = open(results_path + "/best models.txt", "w")

    log_write = open(results_path + "/log.txt", "w")

    columns = ['Subject', 'Train_No', 'Accuracy', 'Kappa', 'Precision', 'Recall', 'F1_Score']
    performance_df = pd.DataFrame(columns=columns)

    n_sub = dataset_conf.get('n_sub')
    data_path = dataset_conf.get('data_path')
    isStandard = dataset_conf.get('isStandard')
    batch_size = train_conf.get('batch_size')
    epochs = train_conf.get('epochs')
    patience = train_conf.get('patience')
    lr = train_conf.get('lr')
    LearnCurves = train_conf.get('LearnCurves')
    n_train = train_conf.get('n_train')
    model_name = train_conf.get('model')

    acc = np.zeros((n_sub, n_train))
    kappa = np.zeros((n_sub, n_train))

    for sub in range(n_sub):
        if sub in [1, 2, 3, 5, 6]:
            continue
        in_sub = time.time()
        print('\nTraining on subject ', sub + 1)
        log_write.write('\nTraining on subject  ' + str(sub + 1) + '\n')

        BestSubjAcc = 0
        bestTrainingHistory = []

        X_test, _, y_test_onehot, X_train, _, y_train_onehot = get_data(data_path, sub, isStandard)  # fold 2

        # X_train, _, y_train_onehot, X_test, _, y_test_onehot = get_data(data_path, sub, isStandard)  # fold 1

        for train in range(n_train):
            in_run = time.time()

            filepath = results_path + '/saved models/run-{}'.format(train + 1)
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            filepath = filepath + '/subject-{}.h5'.format(sub + 1)

            model = getModel(model_name)
            model.compile(loss=categorical_crossentropy, optimizer=Adam(learning_rate=lr), metrics=['accuracy'])

            callbacks = [
                ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0,
                                save_best_only=True, save_weights_only=True, mode='max'),
                EarlyStopping(monitor='val_accuracy', verbose=1, mode='max', patience=patience)
            ]

            history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                                epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

            model.load_weights(filepath)
            y_pred = model.predict(X_test).argmax(axis=-1)
            labels = y_test_onehot.argmax(axis=-1)
            acc[sub, train] = accuracy_score(labels, y_pred)
            kappa[sub, train] = cohen_kappa_score(labels, y_pred)

            precision = precision_score(labels, y_pred, average='weighted')
            recall = recall_score(labels, y_pred, average='weighted')
            f1 = f1_score(labels, y_pred, average='weighted')

            new_row = pd.DataFrame([{
                'Subject': sub + 1,
                'Train_No': train + 1,
                'Accuracy': acc[sub, train],
                'Kappa': kappa[sub, train],
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1
            }])

            performance_df = pd.concat([performance_df, new_row], ignore_index=True)

            out_run = time.time()
            info = 'Subject: {}   Train no. {}   Time: {:.1f} m   '.format(sub + 1, train + 1,
                                                                           ((out_run - in_run) / 60))
            info = info + 'Test_acc: {:.4f}   Test_kappa: {:.4f}'.format(acc[sub, train], kappa[sub, train])
            print(info)
            log_write.write(info + '\n')

            if (BestSubjAcc < acc[sub, train]):
                BestSubjAcc = acc[sub, train]
                bestTrainingHistory = history

        best_run = np.argmax(acc[sub, :])
        filepath = '/saved models/run-{}/subject-{}.h5'.format(best_run + 1, sub + 1) + '\n'
        best_models.write(filepath)

        out_sub = time.time()
        info = '----------\n'
        info = info + 'Subject: {}   best_run: {}   Time: {:.1f} m   '.format(sub + 1, best_run + 1,
                                                                              ((out_sub - in_sub) / 60))
        info = info + 'acc: {:.4f}   avg_acc: {:.4f} +- {:.4f}   '.format(acc[sub, best_run], np.average(acc[sub, :]),
                                                                          acc[sub, :].std())
        info = info + 'kappa: {:.4f}   avg_kappa: {:.4f} +- {:.4f}'.format(kappa[sub, best_run],
                                                                           np.average(kappa[sub, :]),
                                                                           kappa[sub, :].std())
        info = info + '\n----------'
        print(info)
        log_write.write(info + '\n')

    out_exp = time.time()
    info = '\nTime: {:.1f} h   '.format((out_exp - in_exp) / (60 * 60))
    print(info)
    log_write.write(info + '\n')

    # Save the performance metrics into an Excel file
    best_performance_df = performance_df.loc[performance_df.groupby('Subject')['Accuracy'].idxmax()]
    best_performance_df.to_excel(results_path + '/best_performance_df.xlsx', index=False)

    best_models.close()
    log_write.close()

def getModel(model_name):
    if (model_name == 'Bi_ACTCNET'):
        model = cnn_model.Bi_ACTCNET(
            # Dataset parameters
            n_classes=2,
            Chans=59,
            Samples=1000
        )
    else:
        raise Exception("'{}' model is not supported yet!".format(model_name))
    return model

def run():
    data_path = r"E:\BCICIV_1calib_1000Hz_mat\total\\"
    results_path = os.getcwd() + r"\results_test_fold2.1"

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    dataset_conf = {'n_classes': 2, 'n_sub': 8, 'n_channels': 59, 'data_path': data_path,
                    'isStandard': True, 'LOSO': False}
    train_conf = {'batch_size': 24, 'epochs': 300, 'patience': 1000, 'lr': 0.001,
                  'LearnCurves': True, 'n_train': 10, 'model': 'Bi_ACTCNET'}

    train(dataset_conf, train_conf, results_path)

if __name__ == "__main__":
    run()
