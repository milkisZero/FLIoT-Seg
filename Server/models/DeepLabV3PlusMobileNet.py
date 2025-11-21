import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    UpSampling2D,
    Concatenate,
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
        """
        SYNTHIA Semantic Segmentation을 위한
        DeepLabv3+ (MobileNetV2 backbone) 모델
        입력: 256x256x3, 출력: num_classes
        """

        inputs = Input(shape=self.input_shape)

        # Encoder: MobileNetV2 backbone
        # 주의: weights=None 으로 두면 다운로드 없음
        backbone = tf.keras.applications.MobileNetV2(
            input_tensor=inputs,
            include_top=False,
            weights=self.backbone_weights,  # None or "imagenet"
        )

        # low-level feature (stride 4 근처)
        # 보통 block_3_expand_relu 를 많이 씀
        low_level = backbone.get_layer("block_3_expand_relu").output

        # high-level feature (stride 16 근처)
        # 보통 block_13_expand_relu 를 사용
        x = backbone.get_layer("block_13_expand_relu").output

        # ASPP
        x = self._aspp_block(x, filters=256, rate_list=(6, 12, 18))

        # 1/16 -> 1/4 업샘플 (256 입력 기준: 16 → 64)
        x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

        # low-level feature 채널 줄이기
        low_level = Conv2D(48, 1, padding="same", use_bias=False)(low_level)
        low_level = BatchNormalization()(low_level)
        low_level = Activation("relu")(low_level)

        # concat (1/4 해상도에서 결합)
        x = Concatenate()([x, low_level])

        # decoder conv
        x = Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        # 원본 해상도(256x256)로 업샘플 (1/4 -> 1)
        x = UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

        # 최종 클래스 예측
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