import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import tempfile
from dotenv import load_dotenv
import pymongo
import faiss
import numpy as np

# Загрузка переменных окружения из .env файла
load_dotenv()

# Инициализация сессионных переменных
if "conversation" not in st.session_state:
    st.session_state.conversation = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Функция для загрузки Конституции Казахстана
def load_constitution():
    url = "https://www.akorda.kz/en/constitution-of-the-republic-of-kazakhstan-50912"

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        main_content = soup.find('div', class_='content-block')

        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            content = soup.get_text(separator=' ', strip=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(content.encode('utf-8'))
            temp_file_path = temp_file.name

            loader = TextLoader(temp_file_path)
            documents = loader.load()

            os.unlink(temp_file_path)

            return documents

    except Exception as e:
        st.error(f"Ошибка при загрузке Конституции: {e}")
        return []

# Функция для обработки загруженных файлов
def process_uploaded_files(uploaded_files):
    documents = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif file.name.endswith('.docx'):
            loader = Docx2txtLoader(temp_file_path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(temp_file_path)
        else:
            st.warning(f"Неподдерживаемый тип файла: {file.name}")
            os.unlink(temp_file_path)
            continue

        try:
            doc = loader.load()
            documents.extend(doc)
            os.unlink(temp_file_path)
        except Exception as e:
            st.error(f"Ошибка при обработке файла {file.name}: {e}")
            os.unlink(temp_file_path)

    return documents

# Функция для создания индекса и настройки разговора
def setup_conversation(documents):
    # Разделяем документы на чанки
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Инициализируем модель эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Настройка подключения к MongoDB
    mongo_conn_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = "constitution_assistant"
    collection_name = "document_embeddings"

    # Создаем клиент MongoDB
    client = pymongo.MongoClient(mongo_conn_string)

    # Проверка и создание коллекции, если ее нет
    if db_name not in client.list_database_names():
        client[db_name].create_collection(collection_name)

    # Создаем FAISS векторное хранилище
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Сохраняем вектор в MongoDB
    collection = client[db_name][collection_name]
    for i, chunk in enumerate(chunks):
        # Используем embed_documents вместо embed
        vectors = embeddings.embed_documents([chunk.page_content])
        vector = vectors[0]  # Получаем первый вектор для текущего куска
        collection.insert_one({"document": chunk.page_content, "vector": vector})

    # Настройка памяти для чата
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Получаем ключ API для Groq
    groq_api_key = os.getenv("GROQ_API_KEY")

    # Инициализируем языковую модель Groq
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192"  # Вы можете изменить на нужную модель
    )

    # Создаем цепочку разговора
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation, vector_store

# Настройка страницы Streamlit
st.set_page_config(page_title="ИИ-ассистент по Конституции Казахстана", layout="wide")
st.title("ИИ-ассистент по Конституции Республики Казахстан")

with st.sidebar:
    st.header("Настройки")

    if st.button("Загрузить Конституцию Казахстана"):
        with st.spinner("Загрузка Конституции..."):
            constitution_documents = load_constitution()
            if constitution_documents:
                st.session_state.conversation, st.session_state.vector_store = setup_conversation(constitution_documents)
                st.success("Конституция успешно загружена!")

    uploaded_files = st.file_uploader("Загрузите дополнительные документы", accept_multiple_files=True)

    if uploaded_files and st.button("Обработать загруженные файлы"):
        with st.spinner("Обработка загруженных файлов..."):
            documents = process_uploaded_files(uploaded_files)

            if st.session_state.vector_store is not None:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)

                st.session_state.vector_store.add_documents(chunks)

                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                groq_api_key = os.getenv("GROQ_API_KEY")

                llm = ChatGroq(
                    temperature=0,
                    groq_api_key=groq_api_key,
                    model_name="llama3-70b-8192"
                )

                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=st.session_state.vector_store.as_retriever(),
                    memory=memory
                )
            else:
                st.session_state.conversation, st.session_state.vector_store = setup_conversation(documents)

            st.success(f"Обработано {len(documents)} документов!")

# Основная часть интерфейса - чат
st.header("Задайте вопрос о Конституции Казахстана")

chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(f"🙋‍♂️ **Вы:** {message}")
        else:
            st.write(f"🤖 **Ассистент:** {message}")

user_question = st.text_input("Введите ваш вопрос:")

if st.button("Отправить") and user_question:
    if st.session_state.conversation is None:
        st.warning("Пожалуйста, сначала загрузите Конституцию или другие документы!")
    else:
        with st.spinner("Поиск ответа..."):
            st.session_state.chat_history.append(user_question)

            response = st.session_state.conversation({"question": user_question})
            answer = response["answer"]

            st.session_state.chat_history.append(answer)

            st.rerun()  # Исправленный вызов

st.markdown("---")
st.markdown("### О приложении")
st.markdown("""  
Этот ИИ-ассистент предназначен для ответов на вопросы связанные с Конституцией Республики Казахстан.  
Вы можете загрузить текст Конституции, а также дополнительные документы для получения более подробных ответов.  

Приложение использует:  
- Модель LLaMa3 через API Groq  
- Векторное хранилище FAISS для эффективного поиска  
- Алгоритм embeddings для семантического сравнения
""")
