Here's the updated README with the specified instructions for starting the server and setting the environment variables:

---

# RAG Extension Major 2024

## Overview

Welcome to **RAG Extension Major 2024**, a cutting-edge web application designed to create and deploy seamless, customizable chatbots using the Retrieval-Augmented Generation (RAG) architecture.

Our application provides an easy way to integrate a human-free AI chat support agent into your website, enhancing customer interaction and streamlining navigation across your web applications.

With our solution, businesses can effortlessly implement a sophisticated AI-driven support system that responds to customer queries, ensuring a smooth and efficient user experience.

## Key Features

- **Scalability**: Built to grow with your needs, accommodating increased traffic and user interactions effortlessly.
- **Customization**: Tailor the chatbot to fit your brand’s voice and specific customer needs.
- **RAG Architecture**: Leverage advanced AI technology for more intelligent and context-aware responses.
- **Easy Integration**: Simple setup process that allows you to add AI support to your website quickly.

## Setup Instructions

### AI Engine (Python)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**:
   Create a `.env` file in the AI Engine directory and add the following variables:
   ```plaintext
   OPENAI_API_KEY=<your_openai_api_key>
   VECTORDDB_URL=<your_vectordb_url>
   ```

4. **Start the server**:
   Use the following command to start the AI Engine server:
   ```bash
   uvicorn main:App --reload
   ```

### Backend Microservice (Node.js)

1. **Navigate to the backend directory**:
   ```bash
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Set up environment variables**:
   Create a `.env` file in the backend directory and add the following variables:
   ```plaintext
   DATABASE_URL=<your_postgresql_database_url>
   PORT=<desired_port>
   ```

4. **Set up PostgreSQL**:
   Ensure that PostgreSQL is installed and running on your system. Create a database as specified in your `DATABASE_URL`.

---


