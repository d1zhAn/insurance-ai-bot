import streamlit as st
import os
import re
from docx import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LCDocument

st.set_page_config(page_title="Страховой Ассистент (Gemini)", page_icon="🛡️")
st.title("🛡️ ИИ-Ассистент по страхованию (Бесплатный)")

google_api_key = st.sidebar.text_input("Введите Google API Key:", type="password")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

    @st.cache_resource
    def init_bot():
        files = [f for f in os.listdir('.') if f.endswith('.docx')]
        if not files:
            st.warning("⚠️ Не найдено .docx файлов")
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
            st.error(f"Ошибка создания хранилища: {e}")
            return None

    try:
        vectorstore = init_bot()
        if vectorstore is None:
            st.stop()

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Задайте вопрос по страхованию..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🔍 Ищу ответ в документах..."):
                    try:
                        # Извлекаем похожие документы
                        docs = vectorstore.similarity_search(prompt, k=5)
                        context = "\n\n".join([d.page_content for d in docs])
                        
                        # Формируем промпт с контекстом
                        system_message = (
                            "Ты — эксперт по страхованию в Казахстане. "
                            "Отвечай ТОЛЬКО на основе контекста, который будет передан ниже. "
                            "Если в контексте нет ответа, скажи: «В документах нет информации по этому вопросу».\n\n"
                            f"Контекст:\n{context}"
                        )
                        # Вызываем LLM, передавая системное сообщение и последний вопрос
                        from langchain_core.messages import SystemMessage, HumanMessage
                        messages = [
                            SystemMessage(content=system_message),
                            HumanMessage(content=prompt)
                        ]
                        answer = llm.invoke(messages).content
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        with st.expander("📚 Источники"):
                            for i, doc in enumerate(docs[:3], 1):
                                src = doc.metadata.get("source", "неизвестно")
                                st.caption(f"Источник {i} — {src}:")
                                st.text(doc.page_content[:300] + "...")
                    except Exception as e:
                        st.error(f"Ошибка получения ответа: {str(e)}")

    except Exception as e:
        st.error(f"Критическая ошибка: {str(e)}")
else:
    st.info("🔑 Введите Google API Key в боковой панели, чтобы начать.")
    st.markdown("""
    ### Как получить ключ:
    1. [Google AI Studio](https://aistudio.google.com/apikey)
    2. Нажмите **Create API Key** и скопируйте ключ
    """)
