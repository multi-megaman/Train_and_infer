import SAN.central_train_SAN as SAN
import CAN.central_train_CAN as CAN



# SanWords = "./data/word.txt"
# SanValImagesPath = "data/Base_soma_subtracao/val/val_images"
# SanValLabelPath = 'data/Base_soma_subtracao/val/val_labels.txt'


# CanWords = "./data/word_can.txt"
# CanValImagesPath = "data/val_image.pkl"
# CanValLabelPath = 'data/Base_soma_subtracao/val/val_labels_subset.txt'

ModelName = "Teste_SAN"
WordsPath = "data/SAN/word.txt"
imageType = "png"
TrainImagesPath = "data/Base_soma_subtracao/train/train_images"
TrainLabelsPath = "data/Base_soma_subtracao/train/train_labels.txt"
TestImagesPath = "data/Base_soma_subtracao/test/test_images"
TestLabelsPath = "data/Base_soma_subtracao/test/test_labels.txt"
ValImagesPath = "data/Base_soma_subtracao/val/val_images"
ValLabelsPath = "data/Base_soma_subtracao/val/val_labels.txt"

experiment = "SAN"


if experiment == "SAN":
    from SAN.data.gen_pkl_resize_for_central_train import Gen_pkl

    train_image_pkl, train_label_pkl = Gen_pkl(images_path=TrainImagesPath,labels_path=TrainLabelsPath,train_or_test="train",image_type=imageType)
    test_image_pkl, test_label_pkl = Gen_pkl(images_path=TestImagesPath,labels_path=TestLabelsPath,train_or_test="test",image_type=imageType)
    
    SAN_param = dict(experiment='SAN', epoches=7, batch_size=2, workers=0,

        optimizer='Adadelta',
        lr=1,
        lr_decay='cosine',
        eps='1e-6',
        weight_decay='1e-4',

        image_width=3200, image_height=400, image_channel=1, dropout=True, dropout_ratio=0.5, relu=True,
        gradient=100, gradient_clip=True, use_label_mask=False,
        train_image_path= train_image_pkl,
        train_label_path= train_label_pkl,
        eval_image_path= test_image_pkl,
        eval_label_path= test_label_pkl,
        word_path= WordsPath,
        encoder={'net': 'DenseNet', 'input_channels': 1, 'out_channels': 684}, resnet={'conv1_stride': 1},
        densenet={'ratio': 16, 'three_layers': True, 'nDenseBlocks': 16,'growthRate': 24, 'reduction': 0.5, 'bottleneck': True, 'use_dropout': True},
        decoder={'net': 'SAN_decoder', 'cell': 'GRU', 'input_size': 64, 'hidden_size': 64},
        attention={'attention_dim': 512, 'attention_ch': 32},
        hybrid_tree={'threshold': 0.5}, optimizer_save=True,
        checkpoint_dir='checkpoints', finetune=False,
        checkpoint='',
        data_augmentation=0,
        log_dir='logs')
    SAN.train_test_SAN_model(params=SAN_param,model_name=ModelName)

elif experiment =="CAN":
    CAN_param = dict(experiment= "CAN",
        epochs= 500, batch_size= 8, workers= 0, train_parts= 1, valid_parts= 1, valid_start= 0, save_start= 0,
        optimizer= "Adadelta",
        lr= 1, lr_decay= "cosine", step_ratio= 10, step_decay= 5, eps= 1e-6, weight_decay= 1e-4,
        beta= 0.9, dropout= True, dropout_ratio= 0.5,
        relu= True, gradient= 100, gradient_clip= True, use_label_mask= False,
        train_image_path= 'datasets/optuna/train_image.pkl',
        train_label_path= 'datasets/optuna/train_labels.txt',
        eval_image_path= 'datasets/optuna/test_image.pkl',
        eval_label_path= 'datasets/optuna/test_labels.txt',
        word_path= 'datasets/word.txt',
        collate_fn= "collate_fn",
        densenet={
            "ratio": 16,
            "nDenseBlocks": 16,
            "growthRate": 24,
            "reduction": 0.5,
            "bottleneck": True,
            "use_dropout": True},

        encoder={
        "input_channel": 1,
        "out_channel": 684},

        decoder={
        "net": "AttDecoder",
        "cell": 'GRU',
        "input_size": 64,
        "hidden_size": 64},

        counting_decoder={
        "in_channel": 684,
        "out_channel": 22},

        attention={
        "attention_dim": 256,
        "word_conv_kernel": 1},

        attention_map_vis_path= 'vis/attention_map',
        counting_map_vis_path= 'vis/counting_map',

        whiten_type= None,
        max_step= 256,

        optimizer_save= False,
        finetune= False,
        checkpoint_dir= 'checkpoints',
        checkpoint= "checkpoints/CAN_2023-05-10-09-43_decoder-AttDecoder/CAN_2023-05-10-09-43_decoder-AttDecoder_WordRate-0.9341_ExpRate-0.3696_33.pth",
        log_dir= 'logs',
        data_augmentation= 100,
        )
