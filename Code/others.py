import re
import numpy as np
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from util.postgres import pg_connection
from util.milvus import mv_connection

def similarity_table_word_weighted(similiarity_table):
    # input unsorted similiarity_table
    # input shape (n_token X n_criteria X n_places)
    # output shape (n_criteria X n_places)
    # 단어 별 어텐션
    word_weights = np.sum(np.sort(similiarity_table,axis=2)[:,:,-20:],axis=2)
    word_weights = np.exp(word_weights)/np.sum(np.exp(word_weights),axis=0)
    new_similarity=np.sum(word_weights[:,:,np.newaxis]*similiarity_table,axis=0)
    
    return new_similarity

def find_ranking(similarity_table, weights=np.array([1,1,1])):
    word_considered_similarity=similarity_table_word_weighted(similarity_table)
    #input shape : 3X3727
    #output shape : (index,similarity score)
    final_similarity=np.sum(np.exp(weights[:,np.newaxis]*(word_considered_similarity)),axis=0)
    
    ranks = sorted(zip(range(len(final_similarity)), final_similarity), key=lambda x: x[1],reverse=True)
    return ranks

def encode_inputs(input):
    ai_model="text-embedding-ada-002"
    my_openai_api_key=''
    embeddings_model = OpenAIEmbeddings(model=ai_model, openai_api_key=my_openai_api_key)
    outputembedding=np.array(embeddings_model.embed_documents(input))
    return outputembedding

def generate_input_words(query, pg,user_id):
    ##CHANGED
    template = """Query: {query}
    Translate the query in English."""
    
    template_2="""Query:{query}
    Extract [when to do, where to go, age, what kind of activity] each in one word from query.
    If there is no answer, leave it as '-'.
    If there are other information to consider, extract these also as seperate lists."""     
  
    prompt = PromptTemplate(template=template, input_variables=["query"])    
    prompt2 = PromptTemplate(template=template_2, input_variables=["query"])    
    
    llm = OpenAI(temperature=0.3, openai_api_key='')
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    llm_chain2 = LLMChain(prompt=prompt2, llm=llm)
    
    answer1 = llm_chain.run(query)
    
    answer2 = llm_chain2.run(answer1)
    answer2=[i.split(':')[-1] for i in answer2.split('\n')[2:]]
    answer2=add_user_info(answer2,pg,user_id) 
    answer2=[i.replace('', '') for i in answer2]
    return answer2

def add_user_info(query,pg, user_id):
    ##CHANGED
    user=pg.index_search_user('user', user_id)
    if '-' in query[0]:
        query[0]='today'
    if '-' in query[1]:
        query[1]='anywhere'
    if '-' in query[2]:
        query[2]=str(user[4][0])
    query=['Female']+query if user[3][0]=='F' else ['Male']+query
    return query


def generate_input_word2(answer):
    ##CHANGED
    template = """Query: {query}
    You are a machine that generates 3 most related words with , in the context of where you would like to go.
    Do not include Cafe in any case.
    Generate 3 words from the input query.
    """
    prompt = PromptTemplate(template=template, input_variables=["query"])    
    llm = OpenAI(temperature=0, openai_api_key='')
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    p = re.compile('[a-zA-Z]')
    
    result=[]
    for i in answer:
        if '-' not in i:
            j = llm_chain.run(i.replace(' ',''))
            j=j.replace('Answer','')

            answer_list =[''.join(p.findall(t)) for t in j.split(',')]
            result+=answer_list
    result=pd.Series(result).drop_duplicates().to_list()
    return result

def user_favorite_calc(pg,mv, user_id = '2023-270762'):
    
    #Get Activity Ids From Postgre
    print("getting user favorites....")
    activity_ids=pg.activity_index_search("favorite_test",user_id)
    return pg.embedding_index_search(activity_ids)
    
    # Get Activity Embedding From Milvus
    # print("getting user favorites embedding....")
    # fav_embedding=mv.find_vector_by_id(activity_ids)
    
    #return fav_embedding
