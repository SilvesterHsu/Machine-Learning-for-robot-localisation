import os
import tensorflow as tf
from slim.deployment import model_deploy

from long_term_mapping.self_awareness import networks
from long_term_mapping.self_awareness.networks import inputs
from long_term_mapping.self_awareness.networks import arg_parsing
from long_term_mapping.self_awareness.networks import metrics


def _run(args):
    network = networks.catalogue[args.network](args)

    deploy_config = _configure_deployment(args.num_gpus)
    sess = tf.Session(config=_configure_session())

    with tf.device(deploy_config.variables_device()):
        global_step = tf.train.create_global_step()

    with tf.device(deploy_config.optimizer_device()):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=args.learning_rate
        )

    '''Inputs'''
    with tf.device(deploy_config.inputs_device()), tf.name_scope('inputs'):
        pipeline = inputs.Pipeline(args, sess)
        examples, labels = pipeline.data
        images = examples['image']

        image_splits = tf.split(
            value=images,
            num_or_size_splits=deploy_config.num_clones,
            name='split_images'
        )
        label_splits = tf.split(
            value=labels,
            num_or_size_splits=deploy_config.num_clones,
            name='split_labels'
        )

    '''Model Creation'''
    model_dp = model_deploy.deploy(
        config=deploy_config,
        model_fn=_clone_fn,
        optimizer=optimizer,
        kwargs={
            'images': image_splits,
            'labels': label_splits,
            'index_iter': iter(range(deploy_config.num_clones)),
            'network': network,
            'is_training': pipeline.is_training
        }
    )

    '''Metrics'''
    train_metrics = metrics.Metrics(
        labels=labels,
        clone_predictions=[clone.outputs['predictions']
                           for clone in model_dp.clones],
        device=deploy_config.variables_device(),
        name='training'
    )
    validation_metrics = metrics.Metrics(
        labels=labels,
        clone_predictions=[clone.outputs['predictions']
                           for clone in model_dp.clones],
        device=deploy_config.variables_device(),
        name='validation',
        padded_data=True
    )
    validation_init_op = tf.group(
        pipeline.validation_iterator.initializer,
        validation_metrics.reset_op
    )
    train_op = tf.group(
        model_dp.train_op,
        train_metrics.trans_update_op,
        train_metrics.rot_update_op
    )

    '''Summaries'''
    with tf.device(deploy_config.variables_device()):
        train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), sess.graph)
        eval_dir = os.path.join(args.model_dir, 'log', 'eval')
        eval_writer = tf.summary.FileWriter(eval_dir, sess.graph)
        #tf.summary.scalar('cosine_dist', train_metrics.cosine_dist)
        tf.summary.scalar('loss', model_dp.total_loss)
        all_summaries = tf.summary.merge_all()

    '''Model Checkpoints'''
    saver = tf.train.Saver(max_to_keep=args.keep_last_n_checkpoints)
    save_path = os.path.join(args.model_dir, 'model.ckpt')

    '''Model Initialization'''
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    last_checkpoint = tf.train.latest_checkpoint(args.model_dir)
    if last_checkpoint:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='simplenet_localization')
        pretrained_saver = tf.train.Saver(var_list=var_list)
        pretrained_saver.restore(sess, last_checkpoint)
        print "Pretrain Model restored from:", last_checkpoint
        pretrained_saver.restore(sess, last_checkpoint)

    starting_step = sess.run(global_step)

    '''Main Loop'''
    for train_step in range(starting_step, args.max_train_steps):
        sess.run(train_op, feed_dict=pipeline.training_data)

        if train_step % int(args.dataset_size / args.batch_size) == 0:
            print('Epoch {:<0}'.format(train_step / int(args.dataset_size / args.batch_size)))
            sess.run(train_metrics.reset_op)

        '''Summary Hook'''
        if train_step % args.summary_interval == 0:
            results = sess.run(
                fetches={'trans_error': train_metrics.trans_error,
                         'rot_error': train_metrics.rot_error,
                         'predictions': train_metrics.clone_predictions,
                         'target': train_metrics.labels,
                         'summary': all_summaries},
                feed_dict=pipeline.training_data
            )
            train_writer.add_summary(results['summary'], train_step)
            print('Train Step {:<5}: translation error {:>.4}, rotation error {:>.4}'.format(train_step, results[
                'trans_error'], results['rot_error']))

        '''Checkpoint Hooks'''
        if train_step % args.checkpoint_interval == 0:
            saver.save(sess, save_path, global_step)

        '''Eval Hook'''
        if train_step % args.validation_interval == 0:
            predictions = []
            targets = []
            while True:
                try:
                    results = sess.run(
                        fetches={'trans_update_op': validation_metrics.trans_update_op,
                                 'rot_update_op': validation_metrics.rot_update_op,
                                 'predictions': validation_metrics.clone_predictions,
                                 'target': validation_metrics.labels},
                        feed_dict=pipeline.validation_data
                    )
                    predictions.append(results['predictions'][0])
                    targets.append(results['target'])

                except tf.errors.OutOfRangeError:
                    break
            results = sess.run({'trans_error': validation_metrics.trans_error, 'rot_error': validation_metrics.rot_error})

            import numpy as np
            #predictions = np.asarray(predictions, dtype=np.float32)
            #targets = np.asarray(targets, dtype=np.float32)

            predictions = np.concatenate((np.reshape(np.array(predictions[:-1], dtype=np.float32), (-1, 6)),
                                          np.array(predictions[-1], dtype=np.float32)), axis=0)
            targets = np.concatenate((np.reshape(np.array(targets[:-1], dtype=np.float32), (-1, 6)), np.array(targets[-1], dtype=np.float32)), axis=0)

            rmse_trans = np.mean((predictions[:, :3]-targets[:, :3])**2)**0.5
            rmse_rots = np.mean((predictions[:, 3:]-targets[:, 3:])**2)**0.5

            print('Evaluation Step {:<5}: translation error {:>.4}, rotation error {:>.4}'
                  .format(train_step, results['trans_error'], results['rot_error']))
            print rmse_trans
            print rmse_rots

            summary = tf.Summary(value=[
                tf.Summary.Value(tag='trans_error', simple_value=results['trans_error']),
                tf.Summary.Value(tag='rot_error', simple_value=results['rot_error']),
            ])
            eval_writer.add_summary(summary, train_step)
            sess.run(validation_init_op)  # Reinitialize dataset and metrics


def _clone_fn(images,
              labels,
              index_iter,
              network,
              is_training):
    clone_index = next(index_iter)
    images = images[clone_index]
    labels = labels[clone_index]

    unscaled_logits = network.build(images, is_training)

    trans_pred, rot_pred = tf.split(unscaled_logits, [3, 3], axis=1)
    trans_target, rot_target= tf.split(labels, [3, 3], axis=1)

    trans_loss = tf.losses.mean_squared_error(labels=trans_target, predictions=trans_pred)
    rot_loss = tf.losses.mean_squared_error(labels=rot_target, predictions=rot_pred)
    loss = trans_loss + rot_loss*10

    return {
        'predictions': unscaled_logits,
        'images': images,
        'trans_loss': trans_loss,
        'rot_loss': rot_loss,
        'loss': loss
    }


def _configure_deployment(num_gpus):
    return model_deploy.DeploymentConfig(num_clones=num_gpus)


def _configure_session():
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=.8)
    return tf.ConfigProto(allow_soft_placement=True,
                          gpu_options=gpu_config)


def run(args=None):
    args = arg_parsing.ArgParser().parse_args(args)
    with tf.Graph().as_default():
        _run(args)


if __name__ == '__main__':
    run()
