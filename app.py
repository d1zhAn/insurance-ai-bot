import streamlit as st
import os
import re
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document as LCDocument

st.set_page_config(page_title="Страховой Ассистент (Gemini)", page_icon="🛡️")
st.title("🛡️ ИИ-Ассистент по страхованию (Бесплатный)")

google_api_key = st.sidebar.text_input("Введите Google API Key:", type="password")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    @st.cache_resource
    def init_bot():
        files = [f for f in os.listdir('.') if f.endswith('.docx')]
        all_docs = []
        for f_name in files:
            try:
                doc = Document(f_name)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                articles = re.split(r'\n(?=[Сс]татья\s+\d+)', text)
                for art in articles:
                    if art.strip():
                        all_docs.append(LCDocument(page_content=art.strip(), metadata={"source": f_name}))
            except Exception as e:
                st.error(f"Ошибка чтения {f_name}: {e}")
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(all_docs, embeddings)
        return vectorstore

    try:
        vectorstore = init_bot()
        
        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key='answer'
            )

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            memory=st.session_state.memory
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Задайте вопрос по страхованию..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = qa_chain.invoke({"question": prompt})
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    except Exception as e:
        st.error(f"Произошла ошибка: {e}")
else:
    st.info("Пожалуйста, введите Google API Key в боковой панели, чтобы начать бесплатно.")
