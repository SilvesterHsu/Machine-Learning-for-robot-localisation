import argparse

class ArgParser(object):
    def __init__(self):
        self.parser = self._create_parser()

    def parse_args(self, args=None):
        args = self.parser.parse_args(args)
        return args

    @staticmethod
    def _create_parser():
        program_name = 'Squeezenet Training Program'
        desc = 'Program for training squeezenet with periodic evaluation.'
        parser = argparse.ArgumentParser(program_name, description=desc)

        parser.add_argument(
            '--model_dir',
            default='/home/kevin/models/global_localization',
            type=str,
            #required=True,
            help='''Output directory for checkpoints and summaries.'''
        )
        parser.add_argument(
            '--train_tfrecord_filepaths',
            default='/home/kevin/data/michigan_gt/exp1/global_localization.tfrecords',
            nargs='+',
            type=str,
            #required=True,
            help='''Filepaths of the TFRecords to be used for training.'''
        )
        parser.add_argument(
            '--validation_tfrecord_filepaths',
            default='/home/kevin/data/michigan_gt/exp2/global_localization.tfrecords',
            nargs='+',
            type=str,
            #required=True,
            help='''Filepaths of the TFRecords to be used for evaluation.'''
        )
        parser.add_argument(
            '--network',
            type=str,
            default='simplenet_localization',
            #required=True,
        )
        parser.add_argument(
            '--target_image_size',
            default=[400, 400],
            nargs=2,
            type=int,
            help='''Input images will be resized to this.'''
        )
        parser.add_argument(
            '--num_gpus',
            default=1,
            type=int,
            #required=True,
        )
        parser.add_argument(
            '--batch_size',
            default=32,
            type=int,
            #required=True
        )
        parser.add_argument(
            '--dataset_size',
            default=2000,
            type=int,  # required=True

        )
        parser.add_argument(
            '--learning_rate', '-l',
            type=float,
            default=0.001,
            help='''Initial learning rate for ADAM optimizer.'''
        )
        parser.add_argument(
            '--batch_norm_decay',
            type=float,
            default=0.999
        )
        parser.add_argument(
            '--weight_decay',
            type=float,
            default=0.0,
            help='''L2 regularization factor for convolution layer weights.
                    0.0 indicates no regularization.'''
        )
        parser.add_argument(
            '--num_input_threads',
            default=4,
            type=int,
            #required=True,
            help='''The number input elements to process in parallel.'''
        )
        parser.add_argument(
            '--shuffle_buffer',
            default=10000,
            type=int,
            #required=True,
            help='''The minimum number of elements in the pool of training data
                    from which to randomly sample.'''
        )
        parser.add_argument(
            '--seed',
            default=1337,
            type=int
        )
        parser.add_argument(
            '--max_train_steps',
            default=10000000,
            type=int
        )
        parser.add_argument(
            '--summary_interval',
            default=100,
            type=int
        )
        parser.add_argument(
            '--checkpoint_interval',
            default=100,
            type=int
        )
        parser.add_argument(
            '--validation_interval',
            default=1000,
            type=int
        )
        parser.add_argument(
            '--keep_last_n_checkpoints',
            default=3,
            type=int
        )
        return parser
