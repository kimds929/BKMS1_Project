import pymilvus
from pymilvus import (connections, FieldSchema, CollectionSchema, DataType, Collection, db, utility, Milvus)
import pandas as pd
import numpy as np
import os

### TBD (23.11.26)
# 1. 필요없는 code들 쳐내기 (Done)
# 2. Milvus에 vectorize 한 Activities feature embeddings를 DB에 저장하는 함수. 만약 없다면 DB 생성. 있다면 row 추가. (Done)
# 3. similarity 높은 row 뽑기 위해 similarity 계산하는 function. 해당 row들 중 top-k(?) 뽑는 쿼리를 수행하는 함수.



class mv_connection():
    def __init__(self):
        # Pre-Connection Info
        self.__TEAM_PREFIX = 'team11'
        self.__HOST = '147.47.200.145'
        self.__PORT = '39530'

        # Indexing & Searching Configuration
        self.__INDEX_TYPE = 'IVF_FLAT'
        self.__NLIST = 1    # number of cluster units for IVF_FLAT index
        self.__METRIC_TYPE = 'COSINE'     # IP : Inner-Product, COSINE : Cosine Similarity

        # Connection Variables
        self.conn = None

        # Initialize connection
        self.initialize_connection()
        
        # Check if our required collections exist and if not create missing ones.
        print("--------------------------------------------------------------")
        self.create_collection("category")
        self.create_collection("review")
        self.create_collection("else")
        print("--------------------------------------------------------------")
        
    def initialize_connection(self):
        print(f"\nCreate connection to Milvus...")
        try:
            connections.connect(alias=self.__TEAM_PREFIX, host=self.__HOST, port=self.__PORT) 
            self.conn = connections
            # print(self.conn.list_connections())       # List connections        
            # db.list_database(using=self.__TEAM_PREFIX)  # List database
            db.using_database(self.__TEAM_PREFIX + '_db', using = self.__TEAM_PREFIX)   # 사용할 data base 지정                  
            print("\nSuccess in connecting to Milvus!")
            
        except Exception as e:
            print("\nMilvus Error: ", e)
            
    def disconnect(self):
        self.conn.disconnect(self.__TEAM_PREFIX)
        print("Successfully Disconnected Milvus")
        
    def collection(self, collection_name):
        return Collection(name=collection_name, using=self.__TEAM_PREFIX)
    
    def has_collection(self, collection_name):  # 해당 collection이 있는지 체크
        return utility.has_collection(collection_name, using=self.__TEAM_PREFIX)   
    
    def create_collection(self, collection_name, dim=1536):   # 'category', 'review', 'else' 생성 (없을 시)
        # define collection schema ------------------------------------------------------------------------
        if not self.has_collection(collection_name):   # 해당 collection이 없을 시                
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, description="int64",
                            is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector",
                            dim=dim, is_primary=False)
            ]

            schema = CollectionSchema(fields=fields, description=f"Embedding vector of {collection_name}")

            # create collection ---------------------------------------------------------------------------
            Collection(name=collection_name, data=None, schema=schema,
                                    using=self.__TEAM_PREFIX,)          
            print(f"A collection of name \"{collection_name}\" has been created.")
        else:
            print(f"A collection of name \"{collection_name}\" already exists.")

    def drop_collection(self, collection_name):
        if self.has_collection(collection_name):
            collection = Collection(name=collection_name, using=self.__TEAM_PREFIX)
            collection.drop()
            print(f"\n{collection_name} dropped.")
        else:
            print(f"\n{collection_name} does not exist so cannot be dropped.")

    def insert_in_collection(self, collection_name, v_array): # embedding vector를 담은 np.array를 'db_name'이란 이름의 collection에 insert하는 함수
        collection = self.collection(collection_name)
        n_rows = collection.num_entities
        n_vectors = v_array.shape[0]

        # create id column
        ids = np.arange(n_vectors).astype(np.int64) + n_rows
        
        data = [{"id": int(id), "embedding": emb.tolist()} for id, emb in zip(ids, v_array)]

        # insert data
        collection.insert(data=data)
        collection.flush()      # commit (메모리에서 disk에 써짐)
        
        print(f"\nVector data of size {n_vectors} inserted successfully to {collection_name}.") 

    def create_collection_index(self, collection_name, vector_field_name):
        index_param = {
            "index_type": self.__INDEX_TYPE,
            "params": {"nlist": self.__NLIST},
            "metric_type": self.__METRIC_TYPE}
        collection = self.collection(collection_name)
        collection.create_index(vector_field_name, index_param)
        print("\nIndex of collection \"{}\" created:\n{}".format(collection_name, collection.index().params))

    def drop_collection_index(self, collection_name):
        collection = self.collection(collection_name)
        collection.drop_index()
        print("\nIndex of collection \"{}\" dropped.".format(collection_name))
    

    def collection_similarity_multi_vectors(self, collection_name, query_vector):
        search_params = {"metric_type": self.__METRIC_TYPE, 
                         # "params": {"nprobe": 128} 
                         }
        
        query_entity =  query_vector.copy()
        collection = self.collection(collection_name)
        collection.load()
        
        milvus_results = collection.search(data=query_entity, anns_field="embedding",param=search_params, limit = 10000, expr=None)

        results = []
        for qi in range(query_vector.shape[0]):
            results_np = np.array([list(re.to_dict().values())[:2] for re in milvus_results[qi]])
            results_sort=np.array(pd.DataFrame(results_np).sort_values(by=0)[1])
            results.append(results_sort)
        return np.stack(results)
        

    def collection_to_csv(self, collection_name):  # Milvus는 직접적으로 table view를 볼 수 없을 뿐 아니라 대규모 export도 할 수 없음 (하...). 해당 코드는 collection 안에 있는 sample row 체크용
        collection = self.collection(collection_name)
        collection.load()        
        samples = collection.query(expr= "id in [0, 1, 2]", output_fields=["id", "embedding"])  # samples = [{'id': 0, 'embedding': [0.3, ..., 0.9]}, ...]

        samples_ids = [samples[i]['id'] for i in range(len(samples))]
        sample_embeddings = [samples[i]['embedding'] for i in range(len(samples))]
        
        # Create dataframe
        file_name = f"{collection_name}_milvus.csv"
        df = pd.DataFrame({'id': samples_ids, 'embedding': sample_embeddings})
        os.makedirs("retrieved", exist_ok=True)      
        df.to_csv(os.path.join("retrieved", file_name), index=False)
    
        collection.release()
        
    def find_vector_by_id(self, query_ids):
        output_embedding=np.zeros(1536)
        num_query = len(query_ids)
        
        if num_query==0:
            return np.zeros(1536)
        
        for col in ["review","category","else"]:
            collection=self.collection(col)
            collection.load()
            embeddings = collection.query(expr= "id in "+str(query_ids), output_fields=["embedding"])
            embeddings = np.array([embeddings[i]['embedding'] for i in range(len(embeddings))])      
            embeddings = np.sum(embeddings,axis=0)
            collection.release()
            output_embedding=output_embedding+embeddings
        return output_embedding/(num_query)
        
