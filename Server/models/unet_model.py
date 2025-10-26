import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, 
    concatenate, Activation
)
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseGlobalModel
from config.settings import ModelConfig
import json


with open("config.json", "r") as f:
    config = json.load(f)
num_classes, selected_labels = config["num_classes"], config["selected_labels"]

class UNetGlobalModel(BaseGlobalModel):
    def __init__(self):
        super(UNetGlobalModel, self).__init__()

    def build_model(self):
        """
        SYNTHIA Semantic Segmentation을 위한 U-Net 모델
        """
        
        inputs = Input((256, 256, 3))
        
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
        outputs = Conv2D(num_classes, 1, activation='softmax')(conv7)
        
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
        
        print(f"[서버] SYNTHIA용 U-Net 모델 생성 완료 - 입력: (224,224,3), 출력: {num_classes}개 클래스")
        model.summary()
        return model