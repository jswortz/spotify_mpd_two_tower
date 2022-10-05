from datetime import datetime
import sys

# setup
PROJECT_ID = 'hybrid-vertex'
BUCKET_NAME = 'spotify-beam-v3' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
VERSION = sys.argv[5]

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
JOB_NAME = f'spotify-bq-tfrecords-{VERSION}-{TIMESTAMP}'
MAX_WORKERS = '20'
RUNNER = 'DataflowRunner'
NETWORK = 'ucaip-haystack-vpc-network'

BQ_TABLE = 'train_flatten'
# Source data
if len(sys.argv) > 1:
    BQ_TABLE = sys.argv[1]


BQ_DATASET = 'mdp_eda_test'
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds

# storage
ROOT = f'gs://{BUCKET_NAME}/{VERSION}'

DATA_DIR = ROOT + '/data/' # Location to store data
STATS_DIR = ROOT +'/stats/' # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/' # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/' # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
CANDIDATE_DIR = ROOT + "/candidates/"

QUERY = f"SELECT * FROM {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"


# total_samples = 65_346_428  
# samples_per_file = 12_800 
# NUM_TF_RECORDS = total_samples // samples_per_file

total_mb_train = sys.argv[4]
target_shard_size_mb = sys.argv[3]

NUM_TF_RECORDS = int(total_mb_train) // int(target_shard_size_mb)


# if NUM_TF_RECORDS % total_samples:
#     NUM_TF_RECORDS += 1


args = {
    'job_name': JOB_NAME,
    'runner': RUNNER,
    'source_query': QUERY,
    'bq_source_table': TABLE_SPEC,
    'network': NETWORK,
    'candidate_sink': CANDIDATE_DIR,
    'num_tfrecords': NUM_TF_RECORDS,
    'project': PROJECT_ID,
    'region': REGION,
    'staging_location': STAGING_DIR,
    'temp_location': TEMP_DIR,
    'save_main_session': True,
    'version': VERSION,
    'setup_file': './setup.py',
    'folder': sys.argv[2], ## train or valid
}

print("Number of Expected TFRecords: {}".format(NUM_TF_RECORDS)) # 5343


def main():
    from train_pipeline import train_pipe_shape
    train_pipe_shape.run(args)
    
if __name__ == '__main__':
    main()