import psycopg2
from psycopg2 import sql
import pandas as pd
import numpy as np
### TBD (23.11.16.)
# 1. Activities DB에서 row (혹은 Information) 뽑아내는 함수.
# 2. User DB에서 Feature 뽑아내는 함수. (1과 동일한 함수 사용할 수도 있고 따로 구현할 수도 있고.)
# 3. Feedback learning으로 얻은 정보를 User DB에 추가하는 함수.

# Server connection info

# Schema (TBD)

# Initialize connection

class pg_connection():
    def __init__(self):
        # Pre-Connection Info
        self.__HOST = '147.47.200.145'
        self.__DBNAME = 'teamdb11'
        self.__USER = 'team11'
        self.__PASSWORD = 'luckytoday'
        self.__PORT = '34543'
        
        # Connection Variables
        self.conn = None
        self.cursor = None
        
        # Initialize connection
        self.initialize_connection()
        
    def initialize_connection(self):    
        print(f"\nCreate connection to Postgres...")
        try:
            connection_info = "host={} dbname={} user={} password={} port={}".format(
                self.__HOST, self.__DBNAME, self.__USER, self.__PASSWORD, self.__PORT)
            self.conn = psycopg2.connect(connection_info)
            self.cursor = self.conn.cursor()
            print("\nSuccess in connecting to Postgres!")
        except psycopg2.Error as e:
            print("\nPostgres Error: ", e)
            
    def disconnect(self):
        if self.conn != None:
            self.cursor.close()
            self.conn.close()
        print("Successfully Disconnected Postgres")

    # Create or replace table (TBD: Is it really needed?)
    def create_table(self, table_name):
        return "something"

    # Fetch table of the table name as Pandas dataframe
    def fetch_table(self, table_name, columns="*"):
        sql_query = "SELECT {} FROM {};".format(columns, table_name)
        df = pd.read_sql(sql_query, self.conn)
        return df
    
    def index_search(self, table_name, ids):
        sql_query = "SELECT * from {} where id in {} order by array_position(array{},id)".format(table_name, tuple(ids),list(ids))
        self.cursor.execute(sql_query)
        result = self.cursor.fetchall()
        return pd.DataFrame(result)
       

# adr_rev = "./231113 관악구_naver.csv"
# adr_rev_t = "./review_translate_dict.csv"
# adr_cat = "./categories.csv"
# adr_cat_t = "./category_translate_dict.csv"


        with st.form(key="form1"):        
            a = st.checkbox(":star:", value="check", key="favorite_1", on_change=None)
            b = st.toggle("toggle", value="toggle", key="abc", on_change=None)
            print()
            print("aaa ########################################################")
            print(a)
            print(b)
            print("bbb ########################################################")
            print()
            submit = st.form_submit_button(label="btn")