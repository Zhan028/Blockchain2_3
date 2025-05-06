# AI assistant on the Constitution of the RK using Groq API

This project is an MVP (minimum viable product) of an AI assistant that can answer questions related to the Constitution of the Republic of Kazakhstan using the modern Llama 3 language model via Groq API and MongoDB vector store.

## Project Features

- **High-performance language model**: Utilizing Llama 3 (70B parameters) via the Groq API
- **Scalable Storage**: MongoDB for long-term storage of vector document views
- **Chat Interface**: Streamlit-based clear user interface
- **Document Processing**: Uploading and processing files of various formats (PDF, DOCX, TXT)
- **Web Content Retrieval**: Automatic download of the text of the RK Constitution from an official source
- **Vector Search**: Efficient semantic search to identify relevant information

## Requirements

- Python 3.8+
- MongoDB (local installation or remote server)
- API key Groq

## Installation

1. Clone the repository:
   ```bash
   git clone <url repository>
   cd <directory name>
   ```

2. Create and activate the virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate # For Linux/Mac
   venv\Scripts\activate # For Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create an `.env` file based on `.env.advanced`:
   ````bash
   cp .env.advanced .env
   ```

5. Edit the `.env` file and add your API keys and MongoDB settings:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   MONGODB_URI=mongodb://localhost:27017 # Modify as necessary
   ```

## Start the application

````bash
streamlit run app_groq.py
````

Once run, the application will be available in your browser at: http://localhost:8501.

## How to use

1. **Load Constitution**:
   - After launching the application, click the “Download the Constitution of Kazakhstan” button in the sidebar
   - Wait for the download and text processing to complete

2. **Download additional documents** (optional):
   - Drag and drop your files into the download area or select them via explorer
   - Click the “Process Uploaded Files” button
   - The system will automatically extract the text, split it into fragments and save it to vector storage

3. **Ask Questions**:
   - Type your question in the input box and click “Submit”
   - The system will find relevant text snippets and generate an answer using the Llama 3 model
   - Conversation history is saved to provide context to the conversation

## Configuring MongoDB

By default, the application attempts to connect to a local MongoDB installation. If you do not have a local MongoDB installation or want to use a remote server:

1. Install MongoDB locally or access a remote MongoDB server (MongoDB Atlas, etc.)
2. update the `MONGODB_URI` connection string in the `.env` file

For example, for MongoDB Atlas:
```
MONGODB_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<dbname>
```

## Configuring the Groq model

The default model is `llama3-70b-8192`, but you can change it to a different model supported by the Groq API by changing the `model_name` parameter in the `setup_conversation()` function.

Available models (at the time of creation):
- llama3-70b-8192
- llama3-8b-8192
- mixtral-8x7b-32768
- gemma-7b-it

## Advantages of using Groq

- **High Speed**: The Groq API provides exceptionally fast responses thanks to specialized processors
- **Powerful Model**: Llama 3 (70B) provides high quality responses and deep contextual understanding
- **Privacy**: The Llama family models are open source, which provides greater transparency into their operation

## Troubleshooting

**MongoDB connectivity issue**:
- Make sure MongoDB is running and accessible at the specified address
- Verify that you have the necessary access rights
- If using MongoDB Atlas, check the network access rules

**Groq API Errors**:
- Verify that the API key is correct
- Check your Groq account limits
- Verify that the selected model is available in your tariff

**Document processing issues**:
- Check the format of the files you are uploading
- PDF upload problems may require additional dependencies to be installed: `pip install pymupdf`.

## Extended functionality

The project can be easily extended in the following ways:

1. **Support for additional file formats**:
   ```bash
   elif file.name.endswith('.epub'):
       loader = UnstructuredEPubLoader(temp_file_path)
   ```

2. **User Interface Enhancements**:
   - Adding a multi-page interface using `st.sidebar.selectbox`.
   - Visualization of extracted text snippets

3. **Vector Search Optimization**:
   - Customizing search parameters in the `as_retriever()` method
   - Realization of search with filtering by metadata

4. **Multilingual Support**:
   - Adding a language switch and translating the interface
   - Integration with multilingual embeddings models

## Authors

Moldabek Zhanbatyr
Nadir Shugay
Dias Makhatov