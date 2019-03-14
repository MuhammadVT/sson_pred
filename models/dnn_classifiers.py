# Fully Connected Multilayer Perceptron (MLP) with single output
class MLP:
    def __init__(self, input_shape, batch_size=32, n_epochs=100, n_classes=2,
                 loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                 out_dir="./trained_models/MLP/"):

        # Add class attributes
        self.input_shape = input_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir

        # Creat a MLP model
        self.model = self.creat_model()

    def creat_model(self):

        from keras.layers import Input, Conv1D, Dense, Flatten
        from keras.layers import normalization, Activation, pooling
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer
        input_layer = Input(self.input_shape)
        fc_layer = input_layer 

        # Add Dense layer 
        fc_layer = Dense(100, activation="relu")(fc_layer)
        fc_layer = Dropout(0.2, seed=100)(fc_layer)

        # Add Dense layer 
        fc_layer = Dense(50, activation="relu")(fc_layer)
        fc_layer = Dropout(0.2, seed=100)(fc_layer)

######################
#        # Add Dense layer 
#        fc_layer = Dense(100, activation="relu")(fc_layer)
#        fc_layer = Dropout(0.2, seed=100)(fc_layer)
#
#        # Add Dense layer 
#        fc_layer = Dense(100, activation="relu")(fc_layer)
#        fc_layer = Dropout(0.2, seed=100)(fc_layer)
#
#        # Add Dense layer 
#        fc_layer = Dense(100, activation="relu")(fc_layer)
#        fc_layer = Dropout(0.2, seed=100)(fc_layer)
######################

        # Add Dense layer 
        fc_layer = Dense(50, activation="relu")(fc_layer)

        # Output layer
        # Use softmax
        output_layer = Dense(self.n_classes, activation="softmax")(fc_layer)

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


# Fully Convolutional Neural Network (FCNN) with single output
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

        from keras.layers import Input, Conv1D, Dense, Flatten
        from keras.layers import normalization, Activation, pooling
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer
        input_layer = Input(self.input_shape)

        conv_layer = Conv1D(filters=16, kernel_size=3, strides=2, padding="valid")(input_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        # Max pooling layer
        #conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)
        #Dropout
        #conv_layer = Dropout(0.2, seed=100)(conv_layer)

        conv_layer = Conv1D(filters=32, kernel_size=3, strides=2, padding="valid")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        # Max pooling layer
        conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)
#        #Dropout
#        #conv_layer = Dropout(0.2, seed=100)(conv_layer)

        conv_layer = Conv1D(filters=32, kernel_size=3, strides=1, padding="valid")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        # Max pooling layer
        #conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)

        conv_layer = Conv1D(filters=32, kernel_size=3, strides=1, padding="valid")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
#
#        conv_layer = Conv1D(filters=32, kernel_size=3, strides=1, padding="same")(conv_layer)
#        conv_layer = normalization.BatchNormalization()(conv_layer)
#        conv_layer = Activation(activation="relu")(conv_layer)
#        #Dropout
#        #conv_layer = Dropout(0.2, seed=100)(conv_layer)
#
#        conv_layer = Conv1D(filters=32, kernel_size=3, strides=1, padding="same")(conv_layer)
#        conv_layer = normalization.BatchNormalization()(conv_layer)
#        conv_layer = Activation(activation="relu")(conv_layer)

        ## Global pooling layer
        #gap_layer = pooling.GlobalAveragePooling1D()(conv_layer)

        # Max pooling layer
#        conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)

        # Flatten 2D data into 1D
        fc_layer = Flatten()(conv_layer)
        fc_layer = Dropout(0.5, seed=100)(fc_layer)

        # Add Dense layer 
        fc_layer = Dense(50, activation="relu")(fc_layer)
        fc_layer = Dropout(0.2, seed=100)(fc_layer)

        # Add Dense layer 
        fc_layer = Dense(10, activation="relu")(fc_layer)

        # Output layer
        # Use softmax
        output_layer = Dense(self.n_classes, activation="softmax")(fc_layer)

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

# Fully Convolutional Neural Network (FCNN) with multiple outputs
class FCNN_MultiOut:
    def __init__(self, input_shape, batch_size=32, n_epochs=100, n_classes=2,
                 loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                 loss_weights=None,
                 out_dir="./trained_models/FCNN_MultiOut/"):

        # Add class attributes
        self.input_shape = input_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.loss = loss
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir

        # Creat a FCNN_MultiOut model
        self.model = self.creat_model()

    def creat_model(self):

        from keras.layers import Input, Conv1D, Dense
        from keras.layers import normalization, Activation, pooling
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer
        input_layer = Input(self.input_shape, name="main_input")

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
        output_layers = []
        for i in range(self.n_classes):
             output_layers.append(Dense(2, activation="softmax", name="output_"+str(i))(gap_layer))

        # Put all the model components together
        model = Model(inputs=input_layer, outputs=output_layers)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
#        # For TensorBoard visualization
#        log_dir = os.path(out_dir, "logs")
#        TensorBoard(log_dir=log_dir, batch_size=self.batch_size, update_freq='epoch')

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

# ResNet Convolutional Neural Network (ResNet) with single output
class ResNet:
    def __init__(self, input_shape, batch_size=32, n_epochs=100, n_classes=2,
                 n_resnet_units=3,
                 loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                 out_dir="./trained_models/ResNet_MultiOut/"):

        # Add class attributes
        self.input_shape = input_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.n_resnet_units = n_resnet_units
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir

        # Creat a ResNet model
        self.model = self.creat_model()

    def __create_resnet_unit(self, input_layer, n_filters=64, n_layers=3, kernel_sizes=[7, 5, 3],
                             first_resnet_unit=True):

        from keras.layers import Conv1D, add 
        from keras.layers import normalization, Activation
        from keras.layers.core import Dropout

        tmp_layer = input_layer
        for i in range(n_layers):
            conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_sizes[i], padding='same')(tmp_layer)
            conv_layer = normalization.BatchNormalization()(conv_layer)
            if i < n_layers-1:
                conv_layer = Activation('relu')(conv_layer)
                #conv_layer = Dropout(0.2, seed=100)(conv_layer)
            tmp_layer = conv_layer

        # expand the first resnet channels for the sum 
        if first_resnet_unit:
            reslink = Conv1D(filters=n_filters, kernel_size=1, padding='same')(input_layer)
        else:   
            reslink = input_layer 
        reslink = normalization.BatchNormalization()(reslink)

        output_layer = add([reslink, tmp_layer])
        output_layer = Activation('relu')(output_layer)

        return output_layer

    def creat_model(self):

        from keras.layers import Input, Conv1D, Flatten, Dense
        from keras.layers import normalization, Activation, pooling
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer
        input_layer = Input(self.input_shape, name="main_input")

        # CNN layers
        conv_layer = Conv1D(filters=16*1, kernel_size=5, strides=1, padding="valid")(input_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)

        conv_layer = Conv1D(filters=16*1, kernel_size=3, strides=1, padding="valid")(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)

        # Max pooling layer
        conv_layer = pooling.MaxPooling1D(pool_size=2)(conv_layer)


        #conv_layer = input_layer

        #############################
        # ResNet Units
        n_filters = 16 * 1
        n_layers = 3   # Keep it as it is
        #kernel_sizes = [7, 5, 3]   # #elements has to be eqaul to n_layers
        kernel_sizes = [3, 3, 3]   # #elements has to be eqaul to n_layers
        #kernel_sizes = [7, 7, 7]   # #elements has to be eqaul to n_layers
        resnet_unit_input = conv_layer
        for i in range(self.n_resnet_units):
            if i == 0:
                first_resnet_unit=True
            else:
                first_resnet_unit=False
            resnet_unit_output = self.__create_resnet_unit(resnet_unit_input, n_filters=n_filters,
                                                           n_layers=n_layers, kernel_sizes=kernel_sizes,
                                                           first_resnet_unit=first_resnet_unit)
            resnet_unit_input = resnet_unit_output

#        #############################
#        # Max pooling layer
#        conv_layer = pooling.MaxPooling1D(pool_size=2)(resnet_unit_output)
#        #############################
#        # ResNet Units
#        n_filters = 16
#        n_layers = 3
#        #kernel_sizes = [7, 5, 3]   # #elements has to be eqaul to n_layers
#        kernel_sizes = [3, 3, 3]   # #elements has to be eqaul to n_layers
#        resnet_unit_input = conv_layer
#        for i in range(self.n_resnet_units):
#            if i == 0:
#                first_resnet_unit=True
#            else:
#                first_resnet_unit=False
#            resnet_unit_output = self.__create_resnet_unit(resnet_unit_input, n_filters=n_filters,
#                                                           n_layers=n_layers, kernel_sizes=kernel_sizes,
#                                                           first_resnet_unit=first_resnet_unit)
#            resnet_unit_input = resnet_unit_output
#        #############################
#        # Max pooling layer
#        conv_layer = pooling.MaxPooling1D(pool_size=2)(resnet_unit_output)
#        #############################
#        # ResNet Units
#        n_filters = 32
#        n_layers = 3
#        #kernel_sizes = [7, 5, 3]   # #elements has to be eqaul to n_layers
#        kernel_sizes = [3, 3, 3]   # #elements has to be eqaul to n_layers
#        resnet_unit_input = conv_layer
#        for i in range(self.n_resnet_units):
#            if i == 0:
#                first_resnet_unit=True
#            else:
#                first_resnet_unit=False
#            resnet_unit_output = self.__create_resnet_unit(resnet_unit_input, n_filters=n_filters,
#                                                           n_layers=n_layers, kernel_sizes=kernel_sizes,
#                                                           first_resnet_unit=first_resnet_unit)
#            resnet_unit_input = resnet_unit_output
#        #############################

        # Max pooling layer
        conv_layer = pooling.MaxPooling1D(pool_size=2)(resnet_unit_output)

        # Global pooling layer
        #fc_layer = pooling.GlobalAveragePooling1D()(resnet_unit_output)

        # Flatten 2D data into 1D
        fc_layer = Flatten()(conv_layer)
        fc_layer = Dropout(0.2, seed=100)(fc_layer)

        # Add Dense layer 
        fc_layer = Dense(100, activation="relu")(fc_layer)
        fc_layer = Dropout(0.1, seed=100)(fc_layer)

        # Output layer
        # Use softmax
        output_layer = Dense(self.n_classes, activation="softmax")(fc_layer)

        # Put all the model components together
        model = Model(inputs=input_layer, outputs=output_layer)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer,
                      metrics=self.metrics) 
        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.val_acc_{val_acc:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=False, period=5)
        
        self.callbacks = [reduce_lr,model_checkpoint]

        return model

# ResNet Convolutional Neural Network (ResNet) with multiple outputs
class ResNet_MultiOut:
    def __init__(self, input_shape, batch_size=32, n_epochs=100, n_classes=2,
                 n_resnet_units=3,
                 loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                 loss_weights=None,
                 out_dir="./trained_models/ResNet_MultiOut/"):

        # Add class attributes
        self.input_shape = input_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.n_resnet_units = n_resnet_units
        self.loss = loss
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir

        # Creat a ResNet model
        self.model = self.creat_model()

    def __create_resnet_unit(self, input_layer, n_filters=64, n_layers=3, kernel_sizes=[7, 5, 3],
                             first_resnet_unit=True):

        from keras.layers import Conv1D, add 
        from keras.layers import normalization, Activation
        from keras.layers.core import Dropout

        tmp_layer = input_layer
        for i in range(n_layers):
            conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_sizes[i], padding='same')(tmp_layer)
            conv_layer = normalization.BatchNormalization()(conv_layer)
            if i < n_layers-1:
                conv_layer = Activation('relu')(conv_layer)
                #conv_layer = Dropout(0.2, seed=100)(conv_layer)
            tmp_layer = conv_layer

        # expand the first resnet channels for the sum 
        if first_resnet_unit:
            reslink = Conv1D(filters=n_filters, kernel_size=1, padding='same')(input_layer)
        else:   
            reslink = input_layer 
        reslink = normalization.BatchNormalization()(reslink)

        output_layer = add([reslink, tmp_layer])
        output_layer = Activation('relu')(output_layer)
        #output_layer = Dropout(0.2, seed=100)(output_layer)

        return output_layer

    def creat_model(self):

        from keras.layers import Input, Conv1D, Dense
        from keras.layers import normalization, Activation, pooling
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer
        input_layer = Input(self.input_shape, name="main_input")

        # ResNet Units
        n_filters = 64
        n_layers = 3
        kernel_sizes = [7, 5, 3]   # #elements has to be eqaul to n_layers
        resnet_unit_input = input_layer
        for i in range(self.n_resnet_units):
            if i == 0:
                first_resnet_unit=True
            else:
                first_resnet_unit=False
            resnet_unit_output = self.__create_resnet_unit(resnet_unit_input, n_filters=n_filters,
                                                           n_layers=n_layers, kernel_sizes=kernel_sizes,
                                                           first_resnet_unit=first_resnet_unit)
            resnet_unit_input = resnet_unit_output

        # Global pooling layer
        gap_layer_main = pooling.GlobalAveragePooling1D()(resnet_unit_output)

        # Output layer
        # Use softmax
        output_layers = []
        n_output_resnet_units = 0
        for i in range(self.n_classes):
            ####################
            # Add resnet layer to each output before global avg pooling
            if n_output_resnet_units > 0:
                resnet_unit_input_j = resnet_unit_input
                for j in range(n_output_resnet_units):
                    resnet_unit_output_j = self.__create_resnet_unit(resnet_unit_input_j, n_filters=n_filters,
                                                                   n_layers=n_layers, kernel_sizes=kernel_sizes,
                                                                   first_resnet_unit=False)

                    resnet_unit_input_j = resnet_unit_output_j
                # Global pooling layer
                gap_layer = pooling.GlobalAveragePooling1D()(resnet_unit_output_j)
            else:
                gap_layer = gap_layer_main

            ####################
            output_layers.append(Dense(2, activation="softmax", name="output_"+str(i))(gap_layer))

        # Put all the model components together
        model = Model(inputs=input_layer, outputs=output_layers)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
#        # For TensorBoard visualization
#        log_dir = os.path(out_dir, "logs")
#        TensorBoard(log_dir=log_dir, batch_size=self.batch_size, update_freq='epoch')

        self.callbacks = [reduce_lr,model_checkpoint]

        return model

#  Use LSTM to encode the input and then use Fully Connected NN to make prediction
class MLSTM_FCN:
    """ Modified based on https://github.com/houshd/MLSTM-FCN/blob/master """
    def __init__(self, input_shape, batch_size=32, n_epochs=100, n_classes=2,
                 loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"],
                 loss_weights=None, out_dir="./trained_models/MLSTM_FCN/"):

        # Add class attributes
        self.input_shape = input_shape
        self.batch_size = batch_size 
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.loss = loss
        self.loss_weights = loss_weights
        self.optimizer = optimizer
        self.metrics = metrics 
        self.out_dir = out_dir

        # Creat a LSTM model
        self.model = self.creat_model()

    def creat_model(self):

        from keras.layers import Input, Conv1D, Dense, LSTM, Masking, Permute
        from keras.layers import normalization, Activation, pooling, concatenate
        from keras.models import Model 
        from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
        from keras.layers.core import Dropout
        import os

        # Input layer 
        input_layer = Input(self.input_shape)

        # Permute the input layer
        input_layer_shuffle = Permute((1,2))(input_layer)

        # Add masking
        input_layer_masked = Masking(mask_value=0.0)(input_layer_shuffle)

        # LSTM
        lstm_layer = LSTM(32)(input_layer_masked)
        # Dropout
        lstm_layer = Dropout(0.1, seed=100)(lstm_layer)


        # Add CNN layer + squeeze-excite block
        conv_layer = Conv1D(filters=32, kernel_size=8, strides=1, padding="same",\
                            kernel_initializer='he_uniform')(input_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = squeeze_excite_block(conv_layer)

        # Add CNN layer + squeeze-excite block
        conv_layer = Conv1D(filters=64, kernel_size=5, strides=1, padding="same",\
                            kernel_initializer='he_uniform')(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)
        conv_layer = squeeze_excite_block(conv_layer)

        # Add CNN layer
        conv_layer = Conv1D(filters=32, kernel_size=3, strides=1, padding="same",\
                            kernel_initializer='he_uniform')(conv_layer)
        conv_layer = normalization.BatchNormalization()(conv_layer)
        conv_layer = Activation(activation="relu")(conv_layer)

        # Global pooling layer
        conv_layer = pooling.GlobalAveragePooling1D()(conv_layer)
        concat_layer = concatenate([lstm_layer, conv_layer])


        # Softmax output layer
        output_layer = Dense(self.n_classes, activation="softmax")(concat_layer)


        # Put all the model components together
        model = Model(inputs=input_layer, outputs=output_layer)

        # configure the model
        model.compile(loss=self.loss, optimizer=self.optimizer,
                      metrics=self.metrics, loss_weights=self.loss_weights)

        # Reduce the learning rate if plateau occurs on the loss curve
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, 
                                      min_lr=0.00001)

        # Save the model at certain checkpoints 
        fname = "weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5"
        file_path = os.path.join(self.out_dir, fname)
        model_checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=False, period=5)
        
        self.callbacks = [reduce_lr,model_checkpoint]

        return model

def squeeze_excite_block(input):
    ''' Create a squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
        k: width factor
    Returns: a keras tensor
    '''
    from keras.layers import pooling, multiply, Reshape, Dense

    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = pooling.GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


def train_model(model, x_train, y_train, x_val, y_val, class_weights=None,
                batch_size=32, n_epochs=10, callbacks=None, shuffle=True):

    from keras.backend import clear_session
    import datetime as dt

    # Train the model
    stime = dt.datetime.now() 
    fit_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, 
                            validation_data=(x_val, y_val), class_weight=class_weights,
                            callbacks=callbacks, shuffle=shuffle)
    etime = dt.datetime.now() 
    training_time = (etime - stime).total_seconds()/60.    # minutes
    print("Training time is {tm} minutes".format(tm=training_time))

    # Test the model on evaluation data
    clear_session()

    return fit_history

