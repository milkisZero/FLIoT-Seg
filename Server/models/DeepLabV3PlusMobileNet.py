import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    UpSampling2D,
    Concatenate,
    Lambda,  # 추가
)
from tensorflow.keras.optimizers import Adam
from models.base_model import BaseGlobalModel


class DeepLabV3PlusMobileNet(BaseGlobalModel):
    def __init__(self, config, lr=1e-4,
                 backbone_weights="imagenet"):
        """
        backbone_weights: None (랜덤 초기화) 또는 "imagenet"
        """
        self.num_classes = config.num_classes
        self.selected_labels = config.selected_labels
        self.input_shape = config.input_shape
        self.lr = lr
        self.backbone_weights = backbone_weights
        super(DeepLabV3PlusMobileNet, self).__init__(config)

    def _aspp_block(self, x, filters=256, rate_list=(6, 12, 18)):
        """
        간단 ASPP: 1x1 conv + 여러 dilation conv 후 concat
        (image pooling branch는 TF/Keras 버전 이슈 피하려고 생략)
        """
        # 1x1 conv branch
        branch1 = Conv2D(filters, 1, padding="same", use_bias=False)(x)
        branch1 = BatchNormalization()(branch1)
        branch1 = Activation("relu")(branch1)

        branches = [branch1]

        # 3x3 atrous conv branches
        for rate in rate_list:
            b = Conv2D(
                filters,
                3,
                padding="same",
                dilation_rate=rate,
                use_bias=False,
            )(x)
            b = BatchNormalization()(b)
            b = Activation("relu")(b)
            branches.append(b)

        x = Concatenate()(branches)
        x = Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        backbone = tf.keras.applications.MobileNetV2(
            input_tensor=inputs,
            include_top=False,
            weights=self.backbone_weights,
        )

        low_level = backbone.get_layer("block_3_expand_relu").output
        x = backbone.get_layer("block_13_expand_relu").output

        # ASPP
        x = self._aspp_block(x, filters=256, rate_list=(6, 12, 18))

        # 1/16 -> 1/4 업샘플
        x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
        
        # low_level feature 처리
        low_level = Conv2D(48, 1, padding="same", use_bias=False)(low_level)
        low_level = BatchNormalization()(low_level)
        low_level = Activation("relu")(low_level)
        
        # ===== Lambda로 동적 크기 맞추기 =====
        def resize_to_match(tensors):
            """x를 low_level의 크기에 맞춤"""
            x_tensor, low_level_tensor = tensors
            target_shape = tf.shape(low_level_tensor)
            return tf.image.resize(
                x_tensor, 
                [target_shape[1], target_shape[2]], 
                method='bilinear'
            )
        
        x = Lambda(resize_to_match)([x, low_level])

        # concat
        x = Concatenate()([x, low_level])

        # decoder conv
        x = Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # 원본 해상도로 업샘플
        x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
        
        # ===== 최종 출력 크기 고정 =====
        def resize_to_input(x_tensor):
            """출력을 입력 크기에 맞춤"""
            return tf.image.resize(
                x_tensor,
                [self.input_shape[0], self.input_shape[1]],
                method='bilinear'
            )
        
        x = Lambda(resize_to_input)(x)

        outputs = Conv2D(self.num_classes, 1, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        # ignore 레이블(255) 처리용 커스텀 손실
        def custom_sparse_categorical_crossentropy(y_true, y_pred):
            # y_true: [B,H,W], y_pred: [B,H,W,C]
            mask = tf.not_equal(y_true, 255)

            # 유효한 픽셀만 선택 (flatten됨)
            y_true_masked = tf.boolean_mask(y_true, mask)
            y_pred_masked = tf.boolean_mask(y_pred, mask)

            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true_masked, y_pred_masked
            )
            return tf.reduce_mean(loss)

        optimizer = Adam(learning_rate=self.lr)
        model.compile(
            loss=custom_sparse_categorical_crossentropy,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        model.summary()
        return model