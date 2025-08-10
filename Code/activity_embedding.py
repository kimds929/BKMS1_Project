import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

class activities():
    ai_model="text-embedding-ada-002"
    my_openai_api_key='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'

    def __init__(self,i_review,i_review_t, i_category,i_category_t):
        self.reviews_original=i_review[['명칭','카테고리','리뷰']].drop_duplicates(subset=['명칭'])
        self.reviews_original["리뷰"]=self.reviews_original['리뷰'].apply(eval)
        self.review_translate=i_review_t.set_index('카테고리')
        self.category = i_category
        self.category_translate = i_category_t.set_index('카테고리')
        self.category_translate = self.category_translate.rename(columns={'Translation':'category'})
        
        self.kunmin =i_review.drop_duplicates(subset=['명칭'])
        self.kunmin["리뷰"]=self.kunmin['리뷰'].apply(eval)
    
    def cosine_similarity(self, array_a, array_b):
        if sum(array_a)==0 or sum(array_b)==0:
            return 0
        return np.dot(array_a,array_b) /(np.linalg.norm(array_a) * (np.linalg.norm(array_b)))
    
    def category_template(self, row):
        age=''
        for i in range(0,5):
            age=age+str(int(10*(i+1)))+'대, ' if row[i]!=0 else age
        head=''
        for i in range(5,8):
            head=head+self.category.columns[i+1]+', ' if row[i]!=0 else head
        return '이것은 '+age[:-2]+'에 적합하고, '+head[:-2]+'가 즐길 수 있어'
    def category_english_embedding(self):
        embeddings_model = OpenAIEmbeddings(model=self.ai_model, openai_api_key=self.my_openai_api_key)
        category_eng_embedding=pd.DataFrame(columns=["category","embedding"])
        category_eng_embedding["category"]=self.category_translate["category"].drop_duplicates().to_list()
        category_eng_embedding["embedding"]=embeddings_model.embed_documents(category_eng_embedding['category'].to_list())
        self.cat_eng_embedding = category_eng_embedding.set_index('category')
    
    def reviews_english_embedding(self):
        embeddings_model = OpenAIEmbeddings(model=self.ai_model, openai_api_key=self.my_openai_api_key)
        reviews_eng_embedding=pd.DataFrame(columns=["review","embedding"])
        reviews_eng_embedding["review"]=self.review_translate["category"].drop_duplicates().to_list()
        reviews_eng_embedding["embedding"]=embeddings_model.embed_documents(reviews_eng_embedding['review'].to_list())
        self.rev_eng_embedding = reviews_eng_embedding.set_index('review')
        
    def process_reviews_original(self):
        #Category to English
        new=pd.merge(left=self.reviews_original, right=self.category, how="left", on="카테고리")
        filtered_data=new[new["필터링"]==1].drop(columns=["필터링"])
        filtered_data=filtered_data.set_index(keys=pd.Index(list(range(len(filtered_data)))),drop=True)
        filtered_data['리뷰'] = filtered_data['리뷰'].apply(lambda x: {k.replace('"',''):0 if v=='' else int(v) for k,v in x.items()})        
        filtered_data['카테고리']= filtered_data['카테고리'].apply(lambda x: self.category_translate.loc[x][0])
        self.review_data_total=filtered_data
        
        new=pd.merge(left=self.kunmin, right=self.category, how="left", on="카테고리")
        filtered_data=new[new["필터링"]==1].drop(columns=["필터링"])
        filtered_data=filtered_data.set_index(keys=pd.Index(list(range(len(filtered_data)))),drop=True)
        filtered_data['리뷰'] = filtered_data['리뷰'].apply(lambda x: {k.replace('"',''):0 if v=='' else int(v) for k,v in x.items()})        
        filtered_data['카테고리']= filtered_data['카테고리'].apply(lambda x: self.category_translate.loc[x][0])
        self.kunmin =filtered_data
        
    
    #여기서부터는 리뷰 프로세싱
    def choose_top_reviews(self, x, max_p=0.9):
        new_dict={}
        for key,val in x.items():
            if len(key)>0:
                new_word=self.review_translate.loc[key][0]
                if new_word not in new_dict:
                    new_dict[new_word]=val
                else:
                    new_dict[new_word]+=val
            else:
                new_dict['kind service']=val
        x=dict(sorted(new_dict.items(),key= lambda k:k[1],reverse=True))
        x_values = np.array(list(x.values()))
        filter_criteria = x_values.cumsum()/x_values.sum() <= max_p
        return [a for a,b in zip(x.keys(),filter_criteria) if b==True]
    
    def reviews_data_distribution_tool(self, rev_list):
        if len(rev_list)==0:
            return np.zeros(1536)
        rev_list=[l for l in rev_list if len(l) > 0]
        temp = np.array(list(map(lambda x: self.rev_eng_embedding.loc[x][0],rev_list)))
        distribution=np.array(range(1,temp.shape[0]+1))[::-1]/sum(range(1,temp.shape[0]+1))
        return np.sum(distribution.reshape(-1,1) *temp,axis=0)
    
    def reviews_data_total_to_embedding(self):
        myreview=self.review_data_total["리뷰"]
        myreview=myreview.apply(lambda x: self.choose_top_reviews(x,max_p=0.9))
        myreview=myreview.apply(self.reviews_data_distribution_tool)
        null_val=sum(myreview)/len(myreview)
        myreview=myreview.apply(lambda x: null_val if (sum(x==0)!=0) else x)
        self.review_final=myreview
        
    #이제부턴 카테고리 프로세싱
    def to_category_final(self):
        embeddings_model = OpenAIEmbeddings(model=self.ai_model, openai_api_key=self.my_openai_api_key)
        
        category_final=pd.DataFrame(columns=['category','cat_emb','els_emb'])
        category=self.category[self.category["필터링"]==1].drop(columns=["필터링"])
        category_final['category']=category['카테고리'].apply(lambda x:self.category_translate.loc[x][0])
        category_final['cat_emb']=category_final['category'].apply(lambda x:self.cat_eng_embedding.loc[x][0])
        category_final['els_emb']=category.drop(['카테고리'],axis=1).apply(self.category_template, axis=1)
        category_final=category_final.drop_duplicates(subset=['category'])
        category_final['els_emb']=embeddings_model.embed_documents(category_final['els_emb'].explode().tolist())
        category_final=category_final.set_index('category')
        self.category_final=category_final
        
    #파이널 합치기
    def final_process(self):
        embedding_final = pd.DataFrame(columns=["category","review","else"])
        embedding_final["review"]=self.review_final
        embedding_final["category"]=self.review_data_total["카테고리"]
        embedding_final["else"]=embedding_final["category"].apply(lambda x: np.array(self.category_final['els_emb'].loc[x]))
        embedding_final["category"]=embedding_final["category"].apply(lambda x: np.array(self.category_final.loc[x]['cat_emb']))
        self.embedding_final=embedding_final
        
    def forward(self):
        self.category_english_embedding()
        self.reviews_english_embedding()
        self.process_reviews_original()
        self.reviews_data_total_to_embedding()
        self.to_category_final()
        self.final_process()
        print("Successfully created embeddings.")
        