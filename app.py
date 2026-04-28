import streamlit as st
import os
import re
from docx import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document as LCDocument

st.set_page_config(page_title="Страховой Ассистент", page_icon="🛡️")
st.title("🛡️ ИИ-Ассистент по страхованию РК")

openai_key = st.sidebar.text_input("Введите OpenAI API Key:", type="password")

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    
    @st.cache_resource
    def init_bot():
        # Загрузка документов
        files = [f for f in os.listdir('.') if f.endswith('.docx')]
        all_docs = []
        for f_name in files:
            doc = Document(f_name)
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            articles = re.split(r'\n(?=[Сс]татья\s+\d+)', text)
            for art in articles:
                if art.strip():
                    all_docs.append(LCDocument(page_content=art.strip(), metadata={"source": f_name}))
        
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        return vectorstore

    try:
        vectorstore = init_bot()
        
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key='answer'
            )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Задайте вопрос по документам..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Прямой вызов цепочки без функции ask()
                response = qa_chain.invoke({"question": prompt})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    except Exception as e:
        st.error(f"Ошибка: {e}")
else:
    st.info("Введите API ключ в боковой панели (слева), чтобы начать.")
