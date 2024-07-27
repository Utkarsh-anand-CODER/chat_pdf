# Gemma Model Document Q&A Chatbot

This project is a Streamlit application that enables users to upload PDF documents and ask questions about their content. The chatbot uses a combination of PDF text extraction, embeddings, and conversational retrieval to provide accurate responses based on the document content.

## Features

- **PDF Text Extraction**: Extracts text from uploaded PDF documents.
- **Text Embeddings**: Converts extracted text into embeddings using Google's Generative AI model.
- **Conversational Retrieval**: Uses embeddings and conversational memory to provide accurate and context-based responses to user questions.
- **Streamlit Interface**: Interactive and user-friendly interface for uploading PDFs and asking questions.

## Technologies Used

- **Python**: Core programming language.
- **Streamlit**: Framework for building the web application.
- **PyPDF2**: Library for PDF text extraction.
- **LangChain**: Framework for building chains for conversational retrieval.
- **Google Generative AI Embeddings**: Model for creating text embeddings.
- **FAISS**: Library for efficient similarity search and clustering of dense vectors.
- **ChatGroq**: LLM used for generating responses.
- **dotenv**: Library for managing environment variables.

## Setup

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Installation

1. **Clone the repository**

    ```bash
    git clone https://github.com/Utkarsh-anand-CODER/chat_pdf.git
    cd chat_pdf
    ```

2. **Create a virtual environment and activate it**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**

    Create a `.env` file in the project root and add the following lines, replacing `your_groq_api_key` and `your_google_api_key` with your actual API keys:

    ```env
    GROQ_API_KEY=your_groq_api_key
    GOOGLE_API_KEY=your_google_api_key
    ```

### Running the Application

1. **Run the Streamlit app**

    ```bash
    streamlit run app.py
    ```

2. **Open your web browser**

    Visit `http://localhost:8501` to access the application.
   ![image](https://github.com/user-attachments/assets/5d76464c-0fac-403c-948c-ec2fabf21531)


## Usage

1. **Upload PDFs**

    Use the sidebar to upload one or multiple PDF files.

2. **Process PDFs**

    Click the "Process PDFs" button to extract text and generate embeddings.

3. **Ask Questions**

    Enter your question in the text input box and click "Ask". The chatbot will respond based on the content of the uploaded documents.

## Directory Structure

