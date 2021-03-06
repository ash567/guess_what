import argparse
import logging
import os

# Replaced because of some pickle error
# from multiprocessing import Pool
from multiprocess import Pool
from distutils.util import strtobool


import tensorflow as tf

from generic.data_provider.iterator import Iterator
from generic.tf_utils.evaluator import Evaluator
from generic.tf_utils.optimizer import create_optimizer
from generic.tf_utils.ckpt_loader import load_checkpoint
from generic.utils.config import load_config
from generic.utils.file_handlers import pickle_dump
from generic.data_provider.image_loader import get_img_builder

from guesswhat.data_provider.guesswhat_dataset import Dataset
from guesswhat.data_provider.questioner_batchifier import QuestionerBatchifier
from guesswhat.data_provider.guesswhat_tokenizer import GWTokenizer
from guesswhat.models.qgen.qgen_lstm_network import QGenNetworkLSTM


if __name__ == '__main__':

    ###############################
    #  LOAD CONFIG
    #############################

    parser = argparse.ArgumentParser('QGen network baseline!')

    parser.add_argument("-data_dir", type=str, help="Directory with data")
    parser.add_argument("-exp_dir", type=str, help="Directory in which experiments are stored")
    parser.add_argument("-config", type=str, help='Config file')
    parser.add_argument("-dict_file", type=str, default="dict.json", help="Dictionary file name")
    parser.add_argument("-img_dir", type=str, help='Directory with images')
    parser.add_argument("-load_checkpoint", type=str, help="Load model parameters from specified checkpoint")
    parser.add_argument("-gpu_ratio", type=float, default=1., help="How many GPU ram is required? (ratio)")
    parser.add_argument("-no_thread", type=int, default=1, help="No thread to load batch")
    parser.add_argument("-continue_exp", type = lambda x: bool(strtobool(x)), default="False", help="Continue previously started experiment?")

    args = parser.parse_args()
    config, exp_identifier, save_path = load_config(args.config, args.exp_dir)
    logger = logging.getLogger()

    ###############################
    #  LOAD DATA
    #############################
    # I still do not know where and when are the words are getting coverted into tokens.

    # Load image the actual images. The type of images depend of the config file [raw, fc7]
    

    # Returns a appropriate loader [This function only does the selection]
    # The Loader is supposed to take the image id. Based on image id, build finds what to load.
    # After we have built, it returns a loader. On loader just call get_image to get the image
    logger.info('Loading images..')
    image_loader = get_img_builder(config['model']['image'], args.img_dir)
    crop_loader = None  # get_img_loader(config, 'crop', args.image_dir)

    # Load data
    logger.info('Loading data..')

    # Loads the actual data which should be next feeded to placeholders
    # dialogs Not yet tokenized
    trainset = Dataset(args.data_dir, "train", image_loader, crop_loader)
    validset = Dataset(args.data_dir, "valid", image_loader, crop_loader)
    testset = Dataset(args.data_dir, "test", image_loader, crop_loader)

    # Load dictionary
    logger.info('Loading dictionary..')
    tokenizer = GWTokenizer(os.path.join(args.data_dir, args.dict_file))

    # Build Network
    logger.info('Building network..')
    network = QGenNetworkLSTM(config["model"], num_words=tokenizer.no_words, policy_gradient=False)

    # Build Optimizer
    logger.info('Building optimizer..')
    optimizer, outputs = create_optimizer(network, config["optimizer"])


    ###############################
    #  START TRAINING
    #############################

    # Load config
    batch_size = config['optimizer']['batch_size']
    no_epoch = config["optimizer"]["no_epoch"]

    # create a saver to store/load checkpoint
    saver = tf.train.Saver()

    # CPU/GPU option
    cpu_pool = Pool(args.no_thread, maxtasksperchild=1000)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_ratio)

    # Changed for tensorboard
    # Building the tf.summary writer
    writer_t = tf.summary.FileWriter(args.exp_dir + '/log/train')
    writer_v = tf.summary.FileWriter(args.exp_dir + '/log/val')

    # Changed for tensorboard
    # Keeping how many minibatches have run
    global_train_step = [0]
    global_valid_step = [0]


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:


        # Returns all the input of the network (The placeholder variables)
        sources = network.get_sources(sess)
        logger.info("Sources: " + ', '.join(sources))

        sess.run(tf.global_variables_initializer())
        start_epoch = load_checkpoint(sess, saver, args, save_path)


        # create training tools
        evaluator = Evaluator(sources, network.scope_name, network=network, tokenizer=tokenizer)
        batchifier = QuestionerBatchifier(tokenizer, sources, status=('success',))

        best_val_loss = 1e5
        for t in range(0, config['optimizer']['no_epoch']):

            logger.info('Epoch {}..'.format(t + 1))

            train_iterator = Iterator(trainset,
                                      batch_size=batch_size, pool=cpu_pool,
                                      batchifier=batchifier,
                                      shuffle=True)
            # Changed for 
            [train_loss, _] = evaluator.process(sess, train_iterator, outputs=outputs + [optimizer] + [network.summary], n_batch = global_train_step, writer = writer_t, mod_val = config["freq"])
            print "The Golbal Train Step is : %d" %(global_train_step[0])

            valid_iterator = Iterator(validset, pool=cpu_pool,
                                      batch_size=batch_size*2,
                                      batchifier=batchifier,
                                      shuffle=False)
            # Changed for tensorboard
            [valid_loss, _] = evaluator.process(sess, valid_iterator, outputs=outputs + [network.summary], n_batch = global_valid_step, writer = writer_v, mod_val = config["freq"] )
            print "The Golbal valid Step is : %d" %(global_valid_step[0])

            logger.info("Training loss: {}".format(train_loss))
            logger.info("Validation loss: {}".format(valid_loss))

            if valid_loss < best_val_loss:
                best_train_loss = train_loss
                best_val_loss = valid_loss
                saver.save(sess, save_path.format('params.ckpt'))
                logger.info("Guesser checkpoint saved...")

                pickle_dump({'epoch': t}, save_path.format('status.pkl'))

        # Load early stopping
        saver.restore(sess, save_path.format('params.ckpt'))
        test_iterator = Iterator(testset, pool=cpu_pool,
                                 batch_size=batch_size*2,
                                 batchifier=batchifier,
                                 shuffle=True)
        [test_loss, _] = evaluator.process(sess, test_iterator, outputs)

        logger.info("Testing loss: {}".format(test_loss))
