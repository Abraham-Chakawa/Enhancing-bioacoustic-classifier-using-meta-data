import tensorflow as tf
from tensorflow.keras import backend, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D, Concatenate
from tensorflow.keras.models import Sequential
import tensorflow as tf
from tensorflow.keras import backend, Input, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import ResNet101V2

class CNNNetwork:

    def custom_CNN_network(self):
        # Branch 1
        cnn_input=Input(shape = (128,76,3))
        feature_extractor = Conv2D(filters = 8, kernel_size = 8, activation = 'relu')(cnn_input)
        feature_extractor = Dropout(rate = 0.3)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size = 4)(feature_extractor)
        feature_extractor = Conv2D(filters = 8, kernel_size = 8, activation = 'relu')(feature_extractor)
        feature_extractor = Dropout(rate=0.3)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size=4)(feature_extractor)
        feature_extractor = Flatten()(feature_extractor)
        # Branch 2
        meta_input = Input(shape=(24))
        meta_ann = Concatenate()([feature_extractor, meta_input]) 
        meta_ann = Dense(units = 64, activation='relu')(meta_ann)
        meta_ann = Dropout(rate=0.3)(meta_ann)
            
        # Model softmax output
        softmax_output=Dense(2, activation = 'softmax')(meta_ann)
        
        # Two intputs and one output
        model = Model(inputs=[cnn_input,meta_input], outputs=softmax_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model

    def Two_branch_CNN_network(self):
        # branch 2
        baseModel = ResNet101V2(weights="imagenet", include_top=False,input_shape = (128,76,3))
        for layer in baseModel.layers:
            layer.trainable = False
        feature_extractor=Flatten()(baseModel.layers[-1].output)
        
        # Branch 2
        meta_input = Input(shape=(24))
        meta_ann = Concatenate()([feature_extractor, meta_input]) 
        meta_ann =tf.keras.layers.Dense(units = 64, activation='relu')(meta_ann)
        meta_ann =tf.keras.layers.Dropout(rate=0.3)(meta_ann)
            
        # Model softmax output
        softmax_output=tf.keras.layers.Dense(2, activation = 'softmax')(meta_ann)
        
        # Two intputs and one output
        model = Model(inputs=[baseModel.inputs,meta_input], outputs=softmax_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        
        return model

    def baseline_model(self):

        # Branch 1
        cnn_input=Input(shape = (128,76,3))
        feature_extractor = Conv2D(filters = 8, kernel_size = 8, activation = 'relu')(cnn_input)
        feature_extractor = Dropout(rate = 0.3)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size = 4)(feature_extractor)
        feature_extractor = Conv2D(filters = 8, kernel_size = 8, activation = 'relu')(feature_extractor)
        feature_extractor = Dropout(rate=0.3)(feature_extractor)
        feature_extractor = MaxPool2D(pool_size=4)(feature_extractor)
        feature_extractor = Flatten()(feature_extractor)
        
        meta_ann = Dense(units = 64, activation='relu')(feature_extractor)
        meta_ann = Dropout(rate=0.3)(meta_ann)
            
        # Model softmax output
        softmax_output=Dense(2, activation = 'softmax')(meta_ann)
        
        # One intput and one output
        model = Model(inputs=cnn_input, outputs=softmax_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

        return model
