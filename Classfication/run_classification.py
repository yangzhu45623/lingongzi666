import os
from datasets.dataset import get_dataset
from datasets.transformer import train_transformer,test_transformer
from utils.util import parse_args
from run.trainer import TrainTrainer
from run.trainer import TestTrainer

def train(args):
    train_transform = train_transformer()
    val_transform = test_transformer()
    Dataset = get_dataset(args.train_type,True)
    train_dataset = Dataset(os.path.join(args.datadir,'train'),img_num=args.img_num,transformer=train_transform,data_type=args.data_type,args=args)
    val_dataset = Dataset(os.path.join(args.datadir,'test'),img_num=args.img_num,transformer=val_transform,data_type=args.data_type,args=args)

    trainer = TrainTrainer(args,train_dataset,val_dataset)
    trainer.train()

def test(args,mode):
    val_transform = test_transformer()
    Dataset = get_dataset(args.train_type,True)
    train_dataset = Dataset(os.path.join(args.datadir,'train'),img_num=args.img_num,transformer=val_transform,data_type=args.data_type,args=args)
    val_dataset = Dataset(os.path.join(args.datadir,'test'),img_num=args.img_num,transformer=val_transform,data_type=args.data_type,args=args)

    trainer = TestTrainer(args,train_dataset,val_dataset)

    if mode in ['train','all']:
        trainer.test('train')
    if mode in ['test','all','val']:
        trainer.test('val')


def main(args):
    if args.train:
        train(args)
    else:
        test(args,args.test_mode)

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    main(args)
