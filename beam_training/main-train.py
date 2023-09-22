from datetime import datetime
import sys

import warnings
warnings.filterwarnings('ignore')

# setup
PROJECT_ID = sys.argv[1]                             # 'hybrid-vertex'
NETWORK = sys.argv[2]                                # 'ucaip-haystack-vpc-network'
REGION = sys.argv[3]                                 # 'us-central1' # Set the region for Dataflow jobs
VERSION = sys.argv[4]
BUCKET_NAME = sys.argv[5]                            # 'spotify-data-regimes'
GCS_SUBFOLDER = sys.argv[6]

TOTAL_MB_DS = sys.argv[7]
TARGET_SHARD_SIZE_MB = sys.argv[8]
NUM_TF_RECORDS = int(TOTAL_MB_DS) // int(TARGET_SHARD_SIZE_MB)

# Source data
BQ_DATASET = sys.argv[9]                             # 'a_spotify_hack'
BQ_TABLE = sys.argv[10]                              # 'train_flat_last_5_v9' 
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
JOB_NAME = f'spotify-bq-tfrecords-{VERSION}-{TIMESTAMP}'
MAX_WORKERS = '40'
RUNNER = 'DataflowRunner'

# storage
ROOT = f'gs://{BUCKET_NAME}/data/{VERSION}'

DATA_DIR = ROOT + '/data/'                          # Location to store data
STATS_DIR = ROOT +'/stats/'                         # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/'                # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/'                     # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
CANDIDATE_DIR = ROOT + "/candidates/"

QUERY = f"SELECT * FROM {PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"


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
    'folder': GCS_SUBFOLDER, # sys.argv[2], ## train or valid
    'bucket_name': BUCKET_NAME,
}

print("Number of Expected TFRecords: {}".format(NUM_TF_RECORDS))


def main():
    from train_pipeline import train_pipe_shape
    # from train_pipeline import train_pipe
    
    train_pipe_shape.run(args)
    # train_pipe.run(args) # sequence example
    
if __name__ == '__main__':
    main()