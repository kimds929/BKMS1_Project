import streamlit as st
import numpy as np
import pandas as pd

from util.postgres import pg_connection
from util.milvus import mv_connection
from prev.activity_embedding import activities as act_emb
from model import model
from util.others import *

import os
import psutil
import folium
from streamlit_folium import st_folium
import json
import geopandas as gpd


def query_generation(input, mv, pg):
    st.cache_data.clear()
    input = generate_input_words(input)
    print(input)
    input += generate_input_word2(input)
    print(input)
    query_vector = encode_inputs(input)
    r_sim = mv.collection_similarity_multi_vectors("review", query_vector)
    c_sim = mv.collection_similarity_multi_vectors("category", query_vector)
    e_sim = mv.collection_similarity_multi_vectors("else", query_vector)
    sim_table = np.stack([r_sim, c_sim, e_sim], axis=1)  # (5,3,3727)
    rank = find_ranking(sim_table,weights=np.array([1,1,1]))
    rank=[item[0] for item in rank]
    result = pg.index_search("activity_final_final", rank[:100]) 
    # names=list(result[1])
    return result



# Insert Background Wall-Paper
def add_bg_from_url():
    st.markdown(
         f"""
        <style>
        .stApp {{
            background-image: url("https://piktochart.com/wp-content/uploads/2023/05/large-157-600x338.jpg");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
         """,
         unsafe_allow_html=True
     )

st.set_page_config(layout="wide")
add_bg_from_url()                   # Background Image
main_container = st.container()


def main():
    
    mv = mv_connection()
    pg = pg_connection()

    ## [Login] ##############################################################################

    
    ### [Side Bar] ##########################################################################
    
    user_id = '2023-276242' # Replace it with a ftn that returns the id logged in.

    with st.sidebar:
        user_info = pg.fetch_table('user', where=f"userid = \'{user_id}\'")
        user_name = user_info['name'].values.tolist()[0]
        user_age = user_info['age'].values.tolist()[0]
        user_address = user_info['address'].values.tolist()[0]
        user_sex = user_info['sex'].values.tolist()[0]
        user_favorites = pg.fetch_table('user_favorite', where=f"userid = \'{user_id}\'")['id'].values.tolist()[-3:]
        user_history = pg.fetch_table('user_history', where=f"userid = \'{user_id}\'")['searched_query'].values.tolist()[-3:]

        # User Info Bar
        st.sidebar.header('User Info')
        st.sidebar.subheader('Name')
        st.sidebar.write(user_name)
        st.sidebar.subheader('Age')
        st.sidebar.write(user_age)
        st.sidebar.subheader('Address')
        st.sidebar.write(user_address)
        st.sidebar.subheader('Sex')
        st.sidebar.write(user_sex)
        
        # Favorite section with expander
        favorites_expander = st.expander('My Favorites', expanded=True)
        with favorites_expander:
            if user_favorites:
                for favorite in user_favorites:
                    favorite_name = pg.fetch_table('activities', where=f"id = \'{favorite}\'")['명칭'].values.tolist()[0]
                    st.write(favorite_name)
            else:
                st.write("No favorites yet.")

        # History section with expander
        history_expander = st.expander('My History', expanded=True)
        with history_expander:
            if user_history:
                for history in user_history:
                    st.write(history)
            else:
                st.write("No history yet.")
            
    ### [main] ##########################################################################
    with main_container:
        # Upper menu bar (if needed) -----------------------------------------------------

        
        # Query -----------------------------------------------------------------------
        
        # text_input="오늘 여자친구랑 데이트 있는데 음식점 추천해줘"
        
        sample = {"sample_question" : "What should I do with my girlfriend tomorrow?"}
        auto_complete = st.toggle("☘️어떻게 질문해야 할지 모르겠나요?   왼쪽 토글을 누르면 예시 질문과 답을 볼 수 있어요!☘️")
        
        with st.form(key="form"):
            text_input = st.text_input(
                label='"어디", "누구랑", "무엇을" 하고 싶은지 자세히 적어주시면 더 정확한 결과를 얻을 수 있어요!', 
                value = sample["sample_question"] if auto_complete else ""
                )
            submit_button = st.form_submit_button(label='Lucky Today!')
            
        if submit_button:
            if not text_input:
                st.error("질문을 입력해 주세요!")
            elif len(text_input) < 5:
                st.error("질문을 조금 더 자세하게 적어주세요!")
            else:
                names = query_generation(text_input, mv, pg)
                st.success("오늘은 이런걸 해보는게 어떨까요? 🥳")
                
                # Map ---------------------------------------------------------------------------------------------------------
                # with st.expander("See Map"):
                    
                pjt_folder = 'D:/DataScience/SNU_23-2/23-2 BKMS1/★Team Project' 
                state_geo = json.load(open(f"{pjt_folder}/map_data/관악구_json.json", encoding='utf-8'))
                # state_geo = json.load(open(f"{pjt_folder}/map_data/서울시_json.json", encoding='utf-8'))
                
                # 이미지
                
                center_loc = [37.4782, 126.9518]
                m = folium.Map(location=center_loc, width='100%', hegith='100%', zoom_start=13.2)
                folium.Choropleth(
                    geo_data=state_geo,
                    # data = df,
                    # columns=['SIG_KOR_NM', 'VALUE'],
                    # key_on='features.properties.SIG_KOR_NM',
                    # fill_color='YlGn',
                    fill_color='skyblue',
                    fill_opacity=0.2,
                    line_weight=2,
                    line_color='darkblue',
                ).add_to(m)
                
                # # Marker ----------------------------------------------------------------------------
                # folium.Marker( center_loc, 
                #                 popup = "관악구", 
                #                 tooltip =  "관악구",
                #                 icon=folium.Icon('black', icon='star')
                #                 ).add_to(m)
                
                df_latlag = pg.fetch_table(table_name='address_lat_lng', columns="*")
                
                names_ = pg.fetch_table(table_name='activities', columns="*")
                # zip(names_.columns, np.arange(names_.shape[1]))
                
                print(type(names))
                print(names.shape)
                print(names)
                
                # names_ = names.copy()
                top3_cat = names_['카테고리'].drop_duplicates().head(3).to_list()
                names_top3cat = names_[names_['카테고리'].isin(top3_cat)]
                
                names_top3cat_top2item = names_top3cat.groupby('카테고리').head(2)
                names_top3cat_top2item['카테고리'] = pd.Categorical(names_top3cat_top2item['카테고리'], categories=top3_cat, ordered=True)
                names_final = names_top3cat_top2item.sort_values('카테고리',axis=0)
                df_latlag_final = df_latlag.set_index('id').loc[names_final['id'].to_list()].reset_index()
                names_merge = pd.merge(left=names_final, right=df_latlag_final, on='id')

                cat_to_color = {cat: color for cat, color in zip(names_merge['카테고리'].cat.categories, ['red','blue','green'])}

                # Show Marker
                for ri, rv in names_merge.iterrows():
                    folium.Marker( [rv['lat'],rv['lng']], 
                                popup = rv['명칭'], 
                                tooltip = f"{rv['명칭']} ({rv['카테고리']})",
                                icon=folium.Icon(cat_to_color[rv['카테고리']], icon='star')
                                ).add_to(m)
                
                # # df = pd.read_csv(f"{pjt_folder}/seoul_sample.csv", encoding="utf-8-sig")
                map_data = st_folium(m, center=center_loc, width='100%', height=400, returned_objects=[])
                    
                        
                #     # Contents -----------------------------------------------------------------------
                col1, col2, col3 = st.columns(3)
                for i in range(6):
                    col1.write(names.loc[i,'id'])
                    col2.write(names.loc[i,'명칭'])
                    col3.write(names.loc[i,'카테고리'])
                


    mv.disconnect()
    pg.disconnect()
           

    
if __name__ == "__main__":
    main()
    
    

    
    