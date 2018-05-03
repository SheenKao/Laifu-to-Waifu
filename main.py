import argparse
import os
import tensorflow as tf
tf.set_random_seed(19)
from model import cyclegan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='faces', help='path of the dataset')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--load_size', dest='load_size', type=int, default=128, help='scale images to this size')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=128, help='then crop to this size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='# of output image channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=100, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--generator', dest='generator', default='resnet', help='generation network (resnet, unet, c92)')
parser.add_argument('--discriminator', dest='discriminator', default='default', help='discriminator network (default, deep)')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')

args = parser.parse_args()



args.checkpoint_dir = 'checkpoint'
args.L1_lambda = 1.0

args.continue_train=True
args.dataset_dir='faces'
args.epoch = 10
args.print_freq=200
args.save_freq=500
# args.lr=.0001
args.batch_size=12
args.which_direction='AtoB'
args.ndf=32
# args.train_size=5000
args.use_resnet=True
args.d_train_multiplier = 1
args.generator = 'c92'
args.vgg = 0 #scale of VGG loss; 0 = no vgg; not implemented yet
args.bn = True #use batch norm;

args.model_name = "{}_gen {}_disc {}_ndf {}_VGG {}_BN".format(args.generator,args.discriminator, args.ndf, args.vgg, "1" if args.bn else "0")


# print(1.9*2488/60/60*4)

# resnet model with deep discriminator
# grad descent optimizer for discriminator
# checkpoint_dir = './resnet_dd'



# def main(_):
args.model_dir = os.path.join('models',args.model_name)
args.checkpoint_dir = os.path.join(args.model_dir,args.checkpoint_dir)
args.sample_dir = os.path.join(args.model_dir,args.sample_dir)
args.test_dir = os.path.join(args.model_dir,args.test_dir)
args.log_dir = os.path.join("./logs",args.model_name)

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)
if not os.path.exists(args.sample_dir):
    os.makedirs(args.sample_dir)
if not os.path.exists(args.test_dir):
    os.makedirs(args.test_dir)
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)
model = cyclegan(sess, args)

# args.phase='test'
if args.phase == 'train':
    model.train(args) 
else:
    model.test(args)

# if __name__ == '__main__':
    # tf.app.run()
# model.save(model.checkpoint_dir,step=10000]

# tv = tf.trainable_variables()
# parameters = [[x.name,x.get_shape().num_elements()] for x in tv]
