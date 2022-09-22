from datetime import datetime

PROJECT_ID = 'hybrid-vertex'
BUCKET_NAME = 'spotify-beam-v3' # 'spotify-tfrecords-blog' # Set your Bucket name
REGION = 'us-central1' # Set the region for Dataflow jobs
VERSION = 'v3'

# Pipeline Params
TIMESTAMP = datetime.utcnow().strftime('%y%m%d-%H%M%S')
JOB_NAME = f'spotify-bq-tfrecords-{VERSION}-{TIMESTAMP}'
MAX_WORKERS = '20'
RUNNER = 'DataflowRunner'
NETWORK = 'ucaip-haystack-vpc-network'

# Source data
BQ_TABLE = 'candidates'
BQ_DATASET = 'mdp_eda_test'
TABLE_SPEC = f'{PROJECT_ID}:{BQ_DATASET}.{BQ_TABLE}' # need " : " between project and ds

# storage
ROOT = f'gs://{BUCKET_NAME}/{VERSION}'

DATA_DIR = ROOT + '/data' # Location to store data
STATS_DIR = ROOT +'/stats/' # Location to store stats 
STAGING_DIR = ROOT + '/job/staging/' # Dataflow staging directory on GCP
TEMP_DIR =  ROOT + '/job/temp/' # Dataflow temporary directory on GCP
TF_RECORD_DIR = ROOT + '/tf-records/'
CANDIDATE_DIR = ROOT + '/candidates'

QUERY = f"SELECT * FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`"

NUM_TF_RECORDS = 8

args = {
    'job_name': JOB_NAME,
    'runner': RUNNER,
    'source_query': QUERY,
    'bq_source_table': TABLE_SPEC,
    'network': NETWORK,
    'candidate_sink': CANDIDATE_DIR,
    'num_candidate_tfrecords': NUM_TF_RECORDS,
    'project': PROJECT_ID,
    'region': REGION,
    'staging_location': STAGING_DIR,
    'temp_location': TEMP_DIR,
    'save_main_session': True,
    'setup_file': './setup.py',
}

def main():
    from bq_to_tfr import candidate_pipeline
    candidate_pipeline.run(args)
    
if __name__ == "__main__":   
    main()