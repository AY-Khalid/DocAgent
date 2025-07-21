
# 🩺 DocAgent: An Intelligent Clinical Assistant  

**Author:** Anidu Yakubu Khalid  
**3mtt cohort:** Cohort 3  
**State:** Edo State  
**Fellow ID:** FE/23/72226259  
**Track:** AI/ML

Visit the complete deployed app here: https://docagent.streamlit.app/  

**DocAgent** is a lightweight AI-powered clinical assistant I built using Streamlit and LangChain. It allows healthcare professionals and researchers to query common medical conditions and receive intelligent, contextually relevant responses based on a curated corpus of 30 common ailments.

This project was created and maintained by **Anidu Yakubu Khalid** as a practical tool for medical support, documentation reference, and exploration of Retrieval-Augmented Generation (RAG) in clinical AI.  



## Features

- **Question Answering**: Ask natural-language clinical questions and get concise, evidence-based answers.  
- **RAG Engine**: Utilizes LangChain + FAISS + HuggingFace Embeddings for fast and accurate document retrieval.  
- **Powered by OpenAI GPT**: Uses `gpt-4o-mini` or similar model via the OpenAI API.  
- **Custom Medical Corpus**: Includes more than 20 .txt documents covering diseases like diabetes, hypertension, HIV/AIDS, COVID-19, malaria, and more, scraped from WHO official website "https://www.who.int/news-room/fact-sheets".  
-  **Streamlit UI**: Intuitive web interface for live interaction.    


## Project Structure

```

docAgent/  
├── app.py                      # Main Streamlit app  
├── .env                        # Contains your OpenAI API key (excluded from Git)  
├── requirements.txt            # All dependencies  
├── scrapy.py                   # Web scrapy using beautifulsoup  
├── README.md                   # This documentation  
├── data/    
│   └── corpus/                 # More than 20 text files representing medical condition knowledge base  
├── vectorstore/               # FAISS index (auto-generated; ignored in deployment)  
├── backend/  
│   ├── rag\_engine.py           # RAG logic with vector store + OpenAI integration  
│   └── loader.py               # File loading and text splitting  

````

---

##  How to Run Locally

### 1. Clone the Repository

```bash  
git clone https://github.com/<your-username>/docAgent.git  
cd docAgent  
````

### 2. Create a `.env` File  

Create a `.env` file in the root of the project and paste your OpenAI API key like this:  

```env  
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  
```

> **Keep this file secret!** Never share your API key publicly.  

### 3. Install Dependencies  

```bash   
pip install -r requirements.txt  
```

### 4. Run the App  

```bash  
streamlit run app.py  
```

---

## Deployment (Streamlit Cloud)  

1. Push your project to a GitHub repository.  
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and connect your GitHub.  
3. Select your repo and branch.  
4. Set `OPENAI_API_KEY` as a **secret variable** in Streamlit Cloud.  
5. Click **Deploy**.  

---

## Files to Ignore on Deployment  

Make sure these are in your `.gitignore`:  

```gitignore  
.env  
vectorstore/  
__pycache__/  
*.pyc  
gpt_api_key.docx  
```

---

## Example Prompt  

```
"The patient has been experiencing fever, persistent dry cough, and shortness of breath for the past 4 days. What could be the likely diagnosis and treatment plan?"  
```  

---  

## Contact  

Created by **Anidu Yakubu Khalid**  
Email: aniduyakubu@gmail.com  
GitHub: https://github.com/AY-Khalid  

---

## License  

This project is for educational and healthcare support purposes.  

