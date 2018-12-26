# Fully Convolutional Neural Network (FCNN)
class FCNN:
    def __init__(self, input_shape, batch_size=32, n_epochs=100, n_classes=2,
                 loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                 out_dir="./trained_models/FCNN/"):

        # Add class attributes
        self.input_shape = input_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir

        # Creat a FCNN model
        self.model = self.creat_model()

    def creat_model(self):

        from keras.layers import Input, Conv1D, Dense
        from keras.layers import normalization, Activation, pooling
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer
        input_layer = Input(self.input_shape)

        # First CNN layer
        conv_layer = Conv1D(filters=64, kernel_size=7, strides=1, padding="same")(input_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = Dropout(0.2, seed=100)(conv_layer)

        conv_layer = Conv1D(filters=64, kernel_size=7, strides=1, padding="same")(input_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = Dropout(0.2, seed=100)(conv_layer)

        # Second CNN layer
        conv_layer = Conv1D(filters=128, kernel_size=5, strides=1, padding="same")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = Dropout(0.2, seed=100)(conv_layer)

        conv_layer = Conv1D(filters=128, kernel_size=5, strides=1, padding="same")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = Dropout(0.2, seed=100)(conv_layer)

        conv_layer = Conv1D(filters=128, kernel_size=5, strides=1, padding="same")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = Dropout(0.2, seed=100)(conv_layer)


        # Third CNN layer
        conv_layer = Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = Dropout(0.2, seed=100)(conv_layer)

        conv_layer = Conv1D(filters=64, kernel_size=3, strides=1, padding="same")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)

        # Global pooling layer
        gap_layer = pooling.GlobalAveragePooling1D()(conv_layer)

        # Output layer
        # Use softmax
        # NOTE: use the following softmax activator for binary classification or
        # multi-classification where classes are mutually exclusive.
        #output_layer = Dense(self.n_classes, activation="softmax")(gap_layer)

        # Use sigmoid
        # NOTE: use the following sigmoid activator for all cases, expecially 
        # multi-classification where classes are NOT mutually exclusive.
        output_layer = Dense(self.n_classes, activation="sigmoid")(gap_layer)

        # Put all the model components together
        model = Model(inputs=input_layer, outputs=output_layer)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.val_acc_{val_acc:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
#        # For TensorBoard visualization
#        log_dir = os.path(out_dir, "logs")
#        TensorBoard(log_dir=log_dir, batch_size=self.batch_size, update_freq='epoch')

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

    def train_model(self, x_train, y_train, x_val, y_val, y_true):

        from keras.backend import clear_session
        import datetime as dt

        # Train the model
        stime = dt.datetime.now() 
        fit_history = self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=self.n_epochs, 
                                     validation_data=(x_val, y_val), callbacks=self.callbacks, shuffle=True)
        etime = dt.datetime.now() 
        training_time = (etime - stime).total_seconds()/60.    # minutes
        print("Training time is {tm} minutes".format(tm=training_time))

        # Test the model on evaluation data
        clear_session()

        return fit_history

