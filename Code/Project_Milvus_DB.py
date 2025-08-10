import numpy as np
import pandas as pd
import random
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility, db

_TEAM_PREFIX = 'team11' # <- Change this to your team number

# Server connection info
_HOST = '147.47.200.145'
_PORT = '39530'

# create a connection ------------------------------------------------------
connections.connect(alias=_TEAM_PREFIX, host=_HOST, port=_PORT)       # Create connection...
print(connections.list_connections())       # List connections:

db.list_database(using=_TEAM_PREFIX)    # db 종류
db.using_database(_TEAM_PREFIX + '_db', using = _TEAM_PREFIX)   # 사용할 data base 지정





# collection list ------------------------------------------------------
utility.list_collections(using=_TEAM_PREFIX)    # collection 정보

# collection info ------------------------------------------------------
collecion_name = 'demo'

collection = Collection(collecion_name, using=_TEAM_PREFIX)
collection.description      # description
collection.schema           # schema
collection.schema.fields    # field 정보, 
collection.num_entities     # data 수
collection.indexes          # index 정보
# collection.properties

# pd.DataFrame(collection.query(expr="id_field>0"))     # Collection data조회
# utility.drop_collection('embedding_else', using=_TEAM_PREFIX)    # drop collection



## Load Embedding DataFrame
# embedding_df = cPickle.load(open("embedding.pkl", 'rb'))
# embedding_df


# define collection schema ------------------------------------------------------------------------
# Const names
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Vector parameters
_DIM = 1536
# _INDEX_FILE_SIZE = 32  # max file size of stored index       # 사용안함

for db_name in ['category', 'reviews', 'else']:
    # Schema define
    fields = [
        FieldSchema(name=_ID_FIELD_NAME, dtype=DataType.INT64, description="int64",
                    is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector",
                    dim=_DIM, is_primary=False)
    ]

    schema = CollectionSchema(fields=fields, description=f"{db_name} embedding vector")

    # create collection ---------------------------------------------------------------------------
    collection = Collection(name=f'embedding_{db_name}', data=None, schema=schema,
                            using=_TEAM_PREFIX, 
                            # properties={"collection.ttl.seconds": 1800}
                            )


# input data ---------------------------------------
# for db_name in ['category', 'reviews', 'else']:
db_name = "else"

collecion_name = f'embedding_{db_name}'
collection = Collection(collecion_name, using=_TEAM_PREFIX)

collection.name
collection.schema.fields
collection.num_entities

if (collection.num_entities == 0):
    num = embedding_df.shape[0]
    # sample data
    data = [
            np.array([i for i in range(num)]),
            np.stack(embedding_df[db_name].to_numpy()),
        ]

    collection.insert(data) # insert
    collection.flush()      # commit (메모리에서 disk에 써짐)
    print("fluch success.")




# create index ---------------------------------------
# ['embedding_else', 'embedding_reviews', 'demo', 'embedding_category']

collection = Collection(f"embedding_{db_name}", using=_TEAM_PREFIX)
collection.name
collection.num_entities
collection.schema.fields
# collection.indexes[0].params

_INDEX_TYPE = 'IVF_FLAT'
_NLIST = num     # number of cluster units for IVF_FLAT index
_METRIC_TYPE = 'COSINE'     # IP : Inner-Product, COSINE : Cosine Similarity

index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE
        }
_VECTOR_FIELD_NAME = 'embedding'
collection.create_index(_VECTOR_FIELD_NAME, index_param)


# load and query ---------------------------------------
collection.load()
pd.DataFrame(collection.query(expr="id_field <= 10", output_fields=[e.name for e in collection.schema.fields]))     # Collection data조회

# relase & data validate ---------------------------------------
collection.release()
embedding_df[db_name].iloc[:11]

connections.disconnect(_TEAM_PREFIX)









