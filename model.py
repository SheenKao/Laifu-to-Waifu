from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        self.d_train_multiplier = args.d_train_multiplier
        self.model_dir = args.model_dir
        self.log_dir = args.log_dir
        self.g_norm= 'batch_norm' if args.bn else 'instance_norm'
        if args.discriminator == 'default':
            self.discriminator = discriminator
        elif args.discriminator == 'deep':
            self.discriminator = discriminator_deep
        else:
            print('invalid discriminator')
        
        if args.generator == 'resnet':
            self.generator = generator_resnet
        elif args.generator == 'unet':
            self.generator = generator_unet
        elif args.generator == 'c92':
            self.generator = generator_c92
        else:
            print('invalid generator')
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B", norm=self.g_norm)
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A", norm=self.g_norm)
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A", norm=self.g_norm)
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B", norm=self.g_norm)

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake))
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))
        self.cycle_loss = self.L1_lambda * (abs_criterion(self.real_A, self.fake_A_) + abs_criterion(self.real_B, self.fake_B_))
        self.g_loss = self.cycle_loss + self.g_loss_a2b + self.g_loss_b2a
            

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.cycle_loss_sum = tf.summary.scalar("cycle_loss", self.cycle_loss)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.cycle_loss_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B", norm=self.g_norm)
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A", norm=self.g_norm)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
#         increased epsilon from 1e-8 to 1e-7
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1, epsilon=1e-7) \
            .minimize(self.d_loss, var_list=self.d_vars,global_step=self.global_step_tensor)
        # self.d_optim = tf.train.GradientDescentOptimizer(self.lr) \
            # .minimize(self.d_loss, var_list=self.d_vars,global_step=self.global_step_tensor)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1, epsilon=1e-7) \
            .minimize(self.g_loss, var_list=self.g_vars,global_step=self.global_step_tensor)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train and self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        
        display_freq = 10 #displays progress every 10 steps
        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            print("Learning Rate: {}".format(lr))
            batch_time = time.time()
            total_iterations=args.epoch*batch_idxs
            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Update G network and record fake outputs
                
                if (idx+1) % self.d_train_multiplier == 0:
                    fake_A, fake_B, _, summary_str_g, global_counter = self.sess.run(
                        [self.fake_A, self.fake_B, self.g_optim, self.g_sum, self.global_step_tensor],
                        feed_dict={self.real_data: batch_images, self.lr: lr})
                    
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])
                    # print("updated generator")
                else:
                    fake_A, fake_B, summary_str_g, global_counter = self.sess.run(
                        [self.fake_A, self.fake_B, self.g_sum, self.global_step_tensor],
                        feed_dict={self.real_data: batch_images, self.lr: lr})
                    
                    [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str_d, global_counter = self.sess.run(
                    [self.d_optim, self.d_sum, self.global_step_tensor],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                
                if (global_counter) % 10 == 0:
                    self.writer.add_summary(summary_str_d, global_counter)
                    self.writer.add_summary(summary_str_g, global_counter)

                counter += 1
                now = time.time()
                total_duration=now-start_time
                completion = (counter-1)/total_iterations
                ETA = time.ctime(start_time+total_duration/completion).split()[3]
                if counter % display_freq == 0:
                    print(("%2.2f{}  Epoch: [%2d] [%4d/%4d] batch_time:%4.2f  duration: %4.2f  ETA: {}" % (
                    completion*100, epoch, idx, batch_idxs,  now-batch_time, total_duration)).format('%',ETA))
                batch_time = now
                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, tf.train.global_step(self.sess,self.global_step_tensor))

                if np.mod(counter, args.save_freq) == 1:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        file_name = "cyclegan.model"
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, file_name),
                        global_step=self.global_step_tensor)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        # model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)
        print(sample_images.shape)
#         real_A -> fake_B -> cycle_A (aka fake_A_)
        fake_A, fake_B, cycle_A, cycle_B = self.sess.run(
            [self.fake_A, self.fake_B, self.fake_A_, self.fake_B_],
            feed_dict={self.real_data: sample_images}
        )
#         save_images(fake_A, [self.batch_size, 1],
#                     './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
#         save_images(fake_B, [self.batch_size, 1],
#                     './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        # save_images(np.concatenate((sample_images[:,:,:,3:6],fake_A),axis=0), [self.batch_size, 2],
        #             './{}/A_{:04d}.jpg'.format(sample_dir,  idx))
        # save_images(np.concatenate((sample_images[:,:,:,0:3],fake_B),axis=0), [self.batch_size, 2],
        #             './{}/B_{:04d}.jpg'.format(sample_dir,  idx))
                    
        save_images(np.concatenate((sample_images[:,:,:,3:6],fake_A, cycle_B),axis=0), [self.batch_size, 3],
                    './{}/A_{:04d}.png'.format(sample_dir,  idx))
        save_images(np.concatenate((sample_images[:,:,:,0:3],fake_B, cycle_A),axis=0), [self.batch_size, 3],
                    './{}/B_{:04d}.png'.format(sample_dir,  idx))
                    
#         save_images(sample_images, [self.batch_size,1], './{}/A_{:02d}_{:04d}_o.jpg'.format(sample_dir, epoch, idx))
#         save_images(sample_images, [self.batch_size,1], './{}/B_{:02d}_{:04d}_o.jpg'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
