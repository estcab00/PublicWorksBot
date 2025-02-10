# PublicWorksBot
This project is a langchain chatbot system that will answer questions regarding different public works in Peru. The system consists of a researcher AI, that will look into the json files and a response generator AI, that will give you the final answer based on what the researcher found.

```plaintext
        +---------------------+
        |     User Query      | <--- The user provides the question and the project as input
        +----------+----------+
                   |
                   V
        +----------+----------+
        |   Researcher AI     | <--- Scans JSON data for relevant public works information
        +----------+----------+
                   |
                   V
        +----------+----------+
        | Response Generator  | <--- Uses research results to generate a coherent answer
        +----------+----------+
                   |
                   V
        +----------+----------+
        |   Final Answer      | <--- Displayed to the user
        +---------------------+
```

## 📚 Structure
- `main.py`: The main code of the app. You can locally run it on a terminal
- **📁 json_data:** Contains the json data of the public projects.
- **📁 views:**
  - `chatbot.py`: Stores a langchain chatbot that will respond to questions regarding public works in Peru.
- `tools.py`: Contains the tools used by the langchain system.

## 💬 Getting Started

### Requirements
You need to make sure you have installed the following modules.
- streamlit
- langchain
- langsmith
- langgraph
- openai
- sentence-transformers
- scikit-learn

```bash
pip install streamlit==1.37.1
pip install langchain==0.3.13
pip install openai==1.57.3
pip install langchain-openai==0.2.12
pip install langgraph==0.2.60
pip install langsmith==0.2.3 
pip install sentence-transformers==3.4.1
pip install scikit-learn==1.6.0
```

### Run it locally
You can run the app locally on any terminal

```bash
streamlit run main.py
```

## 💻 Streamlit app
You can access the app [here](https://publicworksbot.streamlit.app/)    
