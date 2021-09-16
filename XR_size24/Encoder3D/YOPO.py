def YOPO(image_size, num_labels):
    kernel_initializer = keras.initializers.orthogonal()
    bias_initializer = keras.initializers.zeros()

    input_shape = (image_size, image_size, image_size, 1)

    main_input = Input(shape=input_shape, name='input_1')

    input_0 = main_input

    x = Conv3D(4, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_0)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out1')(x)
    m1 = GlobalAveragePooling3D()(x)

    x = Conv3D(5, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out2')(x)
    m2 = GlobalAveragePooling3D()(x)

    x = Conv3D(6, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out3')(x)
    m3 = GlobalAveragePooling3D()(x)

    x = Conv3D(7, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out4')(x)
    m4 = GlobalAveragePooling3D()(x)

    x = Conv3D(8, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out5')(x)
    m5 = GlobalAveragePooling3D()(x)

    x = Conv3D(9, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x1 = BatchNormalization(name='conv_out6')(x)
    m6 = GlobalAveragePooling3D()(x1)

    x = Conv3D(3, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_0)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out7')(x)
    m7 = GlobalAveragePooling3D()(x)

    x = Conv3D(4, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out8')(x)
    m8 = GlobalAveragePooling3D()(x)

    x = Conv3D(5, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out9')(x)
    m9 = GlobalAveragePooling3D()(x)

    x = Conv3D(6, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x2 = BatchNormalization(name='conv_out10')(x)
    m10 = GlobalAveragePooling3D()(x2)

    x = Conv3D(2, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_0)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out11')(x)
    m11 = GlobalAveragePooling3D()(x)

    x = Conv3D(3, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out12')(x)
    m12 = GlobalAveragePooling3D()(x)

    x = Conv3D(4, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x3 = BatchNormalization(name='conv_out13')(x)
    m13 = GlobalAveragePooling3D()(x3)

    x = Conv3D(1, (7, 7, 7), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_0)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out14')(x)
    m14 = GlobalAveragePooling3D()(x)

    x = Conv3D(2, (7, 7, 7), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x4 = BatchNormalization(name='conv_out15')(x)
    m15 = GlobalAveragePooling3D()(x4)

    input_1 = Concatenate()([x1, x2, x3, x4])

    input_1 = Dropout(0.5)(input_1)

    x = Conv3D(10, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_1)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out16')(x)
    m16 = GlobalAveragePooling3D()(x)

    x = Conv3D(11, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out17')(x)
    m17 = GlobalAveragePooling3D()(x)

    x = Conv3D(12, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out18')(x)
    m18 = GlobalAveragePooling3D()(x)

    x = Conv3D(13, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out19')(x)
    m19 = GlobalAveragePooling3D()(x)

    x = Conv3D(14, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out20')(x)
    m20 = GlobalAveragePooling3D()(x)

    x = Conv3D(15, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out21')(x)
    m21 = GlobalAveragePooling3D()(x)

    x = Conv3D(16, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out22')(x)
    m22 = GlobalAveragePooling3D()(x)

    x = Conv3D(17, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out23')(x)
    m23 = GlobalAveragePooling3D()(x)

    x = Conv3D(18, (3, 3, 3), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out24')(x)
    m24 = GlobalAveragePooling3D()(x)

    x = Conv3D(7, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_1)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out25')(x)
    m25 = GlobalAveragePooling3D()(x)

    x = Conv3D(8, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out26')(x)
    m26 = GlobalAveragePooling3D()(x)

    x = Conv3D(9, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out27')(x)
    m27 = GlobalAveragePooling3D()(x)

    x = Conv3D(10, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out28')(x)
    m28 = GlobalAveragePooling3D()(x)

    x = Conv3D(11, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out29')(x)
    m29 = GlobalAveragePooling3D()(x)

    x = Conv3D(12, (4, 4, 4), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out30')(x)
    m30 = GlobalAveragePooling3D()(x)

    x = Conv3D(5, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_1)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out31')(x)
    m31 = GlobalAveragePooling3D()(x)

    x = Conv3D(6, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out32')(x)
    m32 = GlobalAveragePooling3D()(x)

    x = Conv3D(7, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out33')(x)
    m33 = GlobalAveragePooling3D()(x)

    x = Conv3D(8, (5, 5, 5), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out34')(x)
    m34 = GlobalAveragePooling3D()(x)

    x = Conv3D(3, (7, 7, 7), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_1)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out35')(x)
    m35 = GlobalAveragePooling3D()(x)

    x = Conv3D(4, (7, 7, 7), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out36')(x)
    m36 = GlobalAveragePooling3D()(x)

    x = Conv3D(5, (7, 7, 7), dilation_rate=(1, 1, 1), padding='valid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(x)
    x = ELU()(x)
    x = BatchNormalization(name='conv_out37')(x)
    m37 = GlobalAveragePooling3D()(x)

    m = Concatenate()(
        [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, m22, m23, m24,
         m25, m26, m27, m28, m29, m30, m31, m32, m33, m34, m35, m36, m37])
    fc1 = BatchNormalization(name='fc1')(m)

    m = Dense(256, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(fc1)
    m = ELU()(m)
    fc2 = BatchNormalization(name='fc2')(m)

    #    out1 = PCA_tensor()(m)
    #    m = ELU()(m)
    #    m = BatchNormalization()(m)

    fc2 = Concatenate()([fc1, fc2])

    m = Dense(128, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(fc2)
    m = ELU()(m)
    fc3 = BatchNormalization(name='fc3')(m)

    out = Concatenate()([fc2, fc3])

    m = Dense(num_labels, activation='softmax')(out)

    mod = keras.models.Model(input=main_input, output=m)

    return mod
