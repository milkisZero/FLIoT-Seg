import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, LeakyReLU, Conv1D, BatchNormalization, MaxPooling1D, Activation, GlobalAveragePooling1D,
    Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, Concatenate
)
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseGlobalModel
import json

class UNetGlobalModel(BaseGlobalModel):
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.selected_labels = config.selected_labels
        self.input_shape = config.input_shape
        super(UNetGlobalModel, self).__init__(config)

    def build_model(self):
        """
        SYNTHIA Semantic Segmentation을 위한 U-Net 모델
        """
        
        inputs = Input(self.input_shape)
        
        # Encoder (다운샘플링)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bottom
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        
        # Decoder (업샘플링)
        up5 = UpSampling2D(size=(2, 2))(conv4)
        merge5 = concatenate([conv3, up5], axis=3)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        
        up6 = UpSampling2D(size=(2, 2))(conv5)
        merge6 = concatenate([conv2, up6], axis=3)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        
        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([conv1, up7], axis=3)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        
        # 출력 레이어
        outputs = Conv2D(self.num_classes, 1, activation='softmax')(conv7)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # ignore 레이블 처리를 위한 커스텀 손실 함수
        def custom_sparse_categorical_crossentropy(y_true, y_pred):
            # ignore 레이블 (255)을 마스크 처리
            mask = tf.not_equal(y_true, 255)
            
            # 유효한 픽셀만 선택
            y_true_masked = tf.boolean_mask(y_true, mask)
            y_pred_masked = tf.boolean_mask(y_pred, mask)
            
            # 표준 sparse categorical crossentropy 적용
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
            
            return tf.reduce_mean(loss)
        
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            loss=custom_sparse_categorical_crossentropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        model.summary()
        return model
    
class UNetLite(BaseGlobalModel):
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.selected_labels = config.selected_labels
        self.input_shape = config.input_shape
        super(UNetLite, self).__init__(config)

    def build_model(self):
        """
        SYNTHIA Semantic Segmentation을 위한 U-Net 모델
        """

        dropout_rate = 0.1        
        inputs = Input(self.input_shape)
        
        # Encoder (다운샘플링) - 기존 구조 유지하되 개선
        # Block 1 - 메모리 절약을 위해 64 → 32로 시작
        conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Activation('relu')(conv1)
        conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = Activation('relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        # Block 2
        conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Activation('relu')(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = Activation('relu')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        # Block 3
        conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Activation('relu')(conv3)
        conv3 = Dropout(dropout_rate)(conv3)
        conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Activation('relu')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # Bottleneck (기존 conv4)
        conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Activation('relu')(conv4)
        conv4 = Dropout(dropout_rate * 1.5)(conv4)
        conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Activation('relu')(conv4)
        
        # Decoder (업샘플링) - 기존 구조 개선
        # Block 5 (기존 up5)
        up5 = UpSampling2D(size=(2, 2))(conv4)
        up5 = Conv2D(128, 2, padding='same', kernel_initializer='he_normal')(up5)
        merge5 = Concatenate()([conv3, up5])
        conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge5)
        conv5 = Activation('relu')(conv5)
        conv5 = Dropout(dropout_rate)(conv5)
        conv5 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Activation('relu')(conv5)
        
        # Block 6 (기존 up6)
        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = Conv2D(64, 2, padding='same', kernel_initializer='he_normal')(up6)
        merge6 = Concatenate()([conv2, up6])
        conv6 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Activation('relu')(conv6)
        conv6 = Dropout(dropout_rate * 0.5)(conv6)
        conv6 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = Activation('relu')(conv6)
        
        # Block 7 (기존 up7)
        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = Conv2D(32, 2, padding='same', kernel_initializer='he_normal')(up7)
        merge7 = Concatenate()([conv1, up7])
        conv7 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Activation('relu')(conv7)
        conv7 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = Activation('relu')(conv7)
        
        # 출력 레이어
        outputs = Conv2D(self.num_classes, 1, activation='softmax')(conv7)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # ignore 레이블 처리를 위한 커스텀 손실 함수
        def custom_sparse_categorical_crossentropy(y_true, y_pred):
            # ignore 레이블 (255)을 마스크 처리
            mask = tf.not_equal(y_true, 255)
            
            # 유효한 픽셀만 선택
            y_true_masked = tf.boolean_mask(y_true, mask)
            y_pred_masked = tf.boolean_mask(y_pred, mask)
            
            # 표준 sparse categorical crossentropy 적용
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred_masked)
            
            return tf.reduce_mean(loss)
        
        optimizer = Adam(learning_rate=0.0001)
        model.compile(
            loss=custom_sparse_categorical_crossentropy,
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        model.summary()
        return model
