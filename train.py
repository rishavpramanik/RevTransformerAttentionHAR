import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.model import model_design


def Train(X,y,folds,batch_size,epochs,learning_rate):
    # Define the number of folds
    avg_acc = []
    avg_recall = []
    avg_f1 = []
    avg_prec=[]
    n_class = y.shape[1]
    _,img_rows, img_cols = X.shape
    for i in range(0, len(folds)):
        train_idx = folds[i][0]
        test_idx = folds[i][1]
        X_train = X[train_idx]
        X_test = X[test_idx]

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                    patience=10, min_lr=1e-6)
        checkpoint_filepath = '/content/checkpt'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, 
                                                                        monitor='val_accuracy', mode='max',save_best_only=True)
        model=model_design(img_rows,img_cols,n_class)
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        print(model.summary())
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
        model.fit(X_train, y[train_idx], batch_size=batch_size, epochs=epochs, verbose=2,validation_data=(X_test,y[test_idx]), callbacks=[reduce_lr,model_checkpoint_callback])
        model.load_weights(checkpoint_filepath)
        Y_Prob = model.predict(X_test)
        Y_Pred = np.argmax(Y_Prob, axis=1)
        Y_Test = np.argmax(y[test_idx], axis=1)
        acc_fold = accuracy_score(Y_Test, Y_Pred)
        avg_acc.append(acc_fold)
        recall_fold = recall_score(Y_Test, Y_Pred, average='macro')
        avg_recall.append(recall_fold)
        f1_fold = f1_score(Y_Test, Y_Pred, average='macro')
        avg_f1.append(f1_fold)
        prec = precision_score(Y_Test, Y_Pred, average='macro')
        avg_prec.append(prec)
        print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold ,i+1))
        print('________________________________________________________________')
    print("Avg Prec", np.mean(avg_prec))
    return avg_acc, avg_recall,avg_f1
