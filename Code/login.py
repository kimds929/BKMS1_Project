import numpy as np
from util.postgres import pg_connection
from util.milvus import mv_connection
from util.activity_embedding import activities as act_emb
from model import model
import streamlit as st
from util.others import *
import streamlit as st
import os
import psutil



def main_page(user):
    st.title("Main Page")
    st.write("This is the main page content.")
    st.write(f"Welcome, {user['name'][0]}!")
    
def main():
    mv = mv_connection()
    pg = pg_connection()
    #Log in Page
    
        

    
    ### Create connection to Milvus & Postgres ###

    # mv = mv_connection()
    # pg = pg_connection()
    
    # sample = {"sample_question" : "What should I do with my girlfriend tomorrow?"}
    # auto_complete = st.toggle("☘️어떻게 질문해야 할지 모르겠나요?   왼쪽 토글을 누르면 예시 질문과 답을 볼 수 있어요!☘️")
    
    # with st.form(key="form"):
    #     text_input = st.text_input(
    #     label='"어디", "누구랑", "무엇을" 하고 싶은지 자세히 적어주시면 더 정확한 결과를 얻을 수 있어요!', 
    #     value = sample["sample_question"] if auto_complete else ""
    #     )
    #     submit_button = st.form_submit_button(label='Lucky Today!')
    # if submit_button:
    #     if not text_input:
    #         st.error("질문을 입력해 주세요!")
    #     elif len(text_input) < 5:
    #         st.error("질문을 조금 더 자세하게 적어주세요!")
    #     else:

    #         st.success("오늘은 이런걸 해보는게 어떨까요? 🥳")
    #         col1, col2 = st.columns(2)
    #         for i in range(20):
    #             col1.write("HELLO")
    #             col2.write("HI")
    # exit_app=st.sidebar.button("Close APP")
    # if exit_app:
    #     pid=os.getpid()
    #     p=psutil.Process(pid)
    #     p.terminate()
           
    mv.disconnect()
    pg.disconnect()
    
if __name__ == "__main__":
    main()
    
    

    
    