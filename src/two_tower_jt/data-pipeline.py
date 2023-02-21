    ### outer parallelism
    
#     train_file_path = f'gs://{args.train_dir}/{args.train_dir_prefix}/*.tfrecords'
#     train_files = tf.data.Dataset.list_files(train_file_path, shuffle=None)
#     train_files.cache()
    
#     def make_dataset(shard_index):
#         files = train_files.shard(args.num_data_shards, shard_index)
#         dataset = tf.data.TFRecordDataset(files)
#         return dataset.batch(GLOBAL_BATCH_SIZE)
    
#     indices = tf.data.Dataset.range(args.num_data_shards)

#     train_dataset = indices.interleave(
#         make_dataset,
#         num_parallel_calls=tf.data.AUTOTUNE
#     ).map(
#         tt.parse_tfrecord, 
#         num_parallel_calls=tf.data.AUTOTUNE
#     ).repeat(
#         args.num_epochs
#     ).prefetch(
#         tf.data.AUTOTUNE
#     ).with_options(
#         options
#     )