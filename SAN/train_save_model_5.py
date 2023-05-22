import os
import cv2
from tqdm import tqdm
import time
import torch
import yaml

from dataset import get_dataset
from models.Backbone import Backbone
from training import train, eval

# """config"""
# params = load_config(args.config)

def train_save_SAN_model(params_optuna, model_name):

    params = dict(experiment='SAN', epoches=500, batch_size=params_optuna['batch_size'], workers=0,

                  optimizer='Adadelta',
                  lr=1,
                  lr_decay='cosine',
                  eps='1e-6',
                  weight_decay='1e-4',

                  image_width=200, image_height=200, image_channel=1, dropout=True, dropout_ratio=0.5, relu=True,
                  gradient=100, gradient_clip=True, use_label_mask=False,
                  train_image_path='data/train_image.pkl',
                  train_label_path='data/train_label.pkl',
                  eval_image_path='data/test_image.pkl',
                  eval_label_path='data/test_label.pkl',
                  word_path='data/word.txt',
                  encoder={'net': 'DenseNet', 'input_channels': 1, 'out_channels': 684}, resnet={'conv1_stride': 1},
                  densenet={'ratio': 16, 'three_layers': params_optuna['three_layers'],
                            'nDenseBlocks':  params_optuna['nDenseBlocks'],
                            'growthRate': params_optuna['growthRate'],
                            'reduction': params_optuna['reduction'],
                            'bottleneck': params_optuna['bottleneck'],
                            'use_dropout':  params_optuna['use_dropout'],},
                  decoder={'net': 'SAN_decoder', 'cell': 'GRU', 'input_size': params_optuna['decoder_input_size'], 'hidden_size': 64},
                  attention={'attention_dim': params_optuna['attention_dim'], 'attention_ch': params_optuna['attention_ch']},
                  hybrid_tree={'threshold': 0.5}, optimizer_save=True,
                  checkpoint_dir='checkpoints', finetune=False,
                  checkpoint='',
                  data_augmentation=params_optuna['data_augmentation'],
                  log_dir='logs')

    from models.CNN.densenet import DenseNet
    model_temp = DenseNet(params=params)

    a = torch.zeros((1, 1, 200, 200))
    out = model_temp(a)

    print(out.shape[1])

    # get the output channels parameter
    params['encoder']['out_channels'] = out.shape[1]

    """random seed"""
    # random.seed(params['seed'])
    # np.random.seed(params['seed'])
    # torch.manual_seed(params['seed'])
    # torch.cuda.manual_seed(params['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'

    if not os.path.exists(os.path.join('checkpoints', model_name)):
        os.makedirs(os.path.join('checkpoints', model_name), exist_ok=True)

        file = open(os.path.join('checkpoints', model_name)+"/config.yaml", "w")
        yaml.dump(params, file)
        file.close()

    params['device'] = device

    print('Using device', device)

    train_loader, eval_loader = get_dataset(params)

    # with tqdm(train_loader, total=len(train_loader)) as pbar:
    #     for batch_idx, (images, image_masks, labels, label_masks) in enumerate(pbar):
    #         for batch_i in range(images.numpy().shape[0]):
    #             print(images.numpy()[batch_i,0,:,:].shape)
    #             cv2.imshow('image', images.numpy()[batch_i,0,:,:])
    #             cv2.waitKey()

    model = Backbone(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    model.name = f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_' \
                 f'max_size-{params["image_height"]}-{params["image_width"]}'
    print(model.name)
    model = model.to(device)

    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                          eps=float(params['eps']), weight_decay=float(params['weight_decay']))


    max_eval_expRate = 0
    min_step = 0
    max_train_expRate = 0
    for epoch in range(params['epoches']):

        train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader, writer=None)

        if (epoch+1) >= 5 and ((epoch+1) % 5 == 0):
        #if True:

            eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(params, model, epoch, eval_loader, writer=None)

            if eval_expRate > max_eval_expRate:
                max_eval_expRate = eval_expRate
                max_train_expRate = train_expRate
                min_step = epoch

                filename = f'{os.path.join("checkpoints", model_name)}/{model_name}_{epoch}.pth'
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)

        # stop if no improvement for more than 30 epochs
        # if (epoch+1) >= min_step + 30:
        #     break


    return max_eval_expRate, max_train_expRate


if __name__ == '__main__':

    params_optuna = {'attention_ch': 32, 'attention_dim': 256, 'batch_size': 8, 'bottleneck': True, 'data_augmentation': 100, 'decoder_input_size': 64, 'growthRate': 24, 'nDenseBlocks': 16, 'reduction': 0.5, 'three_layers': True, 'use_dropout': False}

    model_name = 'optuna_model_5_500e'

    train_save_SAN_model(params_optuna, model_name=model_name)
