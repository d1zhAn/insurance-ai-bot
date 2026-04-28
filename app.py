import streamlit as st
import os
import re
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document as LCDocument
import sys

try:
    from langchain.memory import ConversationBufferMemory
except ImportError:
    try:
        from langchain_community.memory import ConversationBufferMemory
    except ImportError:
        st.error("Не удалось импортировать ConversationBufferMemory")
        st.stop()

st.set_page_config(page_title="Страховой Ассистент (Gemini)", page_icon="🛡️")
st.title("🛡️ ИИ-Ассистент по страхованию (Бесплатный)")

google_api_key = st.sidebar.text_input("Введите Google API Key:", type="password")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    @st.cache_resource
    def init_bot():
        files = [f for f in os.listdir('.') if f.endswith('.docx')]
        if not files:
            st.warning("⚠️ Не найдено .docx файлов в директории")
            return None
            
        all_docs = []
        for f_name in files:
            try:
                doc = Document(f_name)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                if not text.strip():
                    continue
                articles = re.split(r'\n(?=[Сс]татья\s+\d+)', text)
                for art in articles:
                    if art.strip():
                        all_docs.append(LCDocument(page_content=art.strip(), metadata={"source": f_name}))
                st.success(f"✅ Загружен: {f_name} ({len(articles)} статей)")
            except Exception as e:
                st.error(f"❌ Ошибка чтения {f_name}: {e}")
        
        if not all_docs:
            st.error("Не удалось загрузить ни одного документа")
            return None
            
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(all_docs, embeddings)
            return vectorstore
        except Exception as e:
            st.error(f"Ошибка создания векторного хранилища: {e}")
            return None

    try:
        vectorstore = init_bot()
        
        if vectorstore is None:
            st.error("Не удалось инициализировать бота. Проверьте наличие .docx файлов.")
            st.stop()
        
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """Ты - эксперт по страхованию в Казахстане. 
            Отвечай на вопросы ТОЛЬКО на основе предоставленного контекста.
            Если в контексте нет ответа, скажи: "В предоставленных документах нет информации по этому вопросу."
            Контекст: {context}"""),
            ("human", "{input}")
        ])
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 5}),
            combine_docs_chain
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
                with st.spinner("🔍 Ищу информацию в документах..."):
                    try:
                        response = retrieval_chain.invoke({"input": prompt})
                        answer = response.get("answer", "Извините, не удалось получить ответ.")
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        if "context" in response and response["context"]:
                            with st.expander("📚 Источники"):
                                for i, doc in enumerate(response["context"][:3], 1):
                                    st.text(f"Источник {i} (из {doc.metadata.get('source', 'неизвестно')}):")
                                    st.text(doc.page_content[:300] + "...")
                    except Exception as e:
                        st.error(f"Ошибка при получении ответа: {str(e)}")

    except Exception as e:
        st.error(f"Критическая ошибка: {str(e)}")
        st.exception(e)
else:
    st.info("🔑 Пожалуйста, введите Google API Key в боковой панели, чтобы начать.")
    st.markdown("""
    ### 📖 Как получить бесплатный API ключ:
    1. Перейдите на [Google AI Studio](https://aistudio.google.com/apikey)
    2. Нажмите **"Create API Key"**
    3. Скопируйте ключ и вставьте его здесь
    
    💰 **Бесплатный лимит:** 15 запросов в минуту  
    📁 **Поддерживаемые файлы:** .docx документы с законами о страховании
    """)
