# DocAgent

> An AI-powered clinical assistant that helps clinicians and patients with symptom triage and next-step guidance. Built with Python, FastAPI, and OpenAI/Gemini LLMs.

**🔗 Live demo:** ([DocAgent](https://docagent.streamlit.app/))  
**👨‍💻 Author:** [Anidu Yakubu Khalid](https://github.com/AY-Khalid)   

---

## What it does  

DocAgent takes a patient's described symptoms in plain language and returns:  

- A structured triage summary (chief complaint, symptom duration, key features)   
- A ranked list of possible conditions to consider, with brief reasoning   
- Suggested next steps (e.g., self-care, see GP within 24 hours, urgent care, emergency)  
- Clear flags when symptoms suggest a possible emergency  

The system is designed as a **support tool for clinicians and a guidance tool for patients in low-resource settings** — not a replacement for medical advice.  

## Why I built it  

Across much of Nigeria and other underserved regions, patients face long wait times and limited access to primary-care clinicians. DocAgent is an experiment in whether an LLM, prompted carefully and with explicit safety rails, can help patients understand whether their symptoms need urgent attention — and help clinicians work through intake faster.  

## How it works  

```
Patient input (free text)  
        │
        ▼
  ┌────────────────┐
  │ Symptom parser │  Extracts structured features (onset, severity, location, etc.)  
  └────────────────┘
        │
        ▼
  ┌────────────────┐
  │  Triage engine │  Calls LLM with a carefully designed clinical prompt  
  └────────────────┘
        │
        ▼
  ┌────────────────┐
  │  Safety layer  │  Flags red-flag symptoms (chest pain, stroke signs, etc.)  
  └────────────────┘
        │
        ▼
Structured triage response + recommended next steps  
```

## Tech stack  

- **Backend:** Python, FastAPI  
- **AI:** OpenAI GPT-4 (primary), Gemini (fallback)  
- **Frontend:** [list your frontend here — Vue / Streamlit / plain HTML, whichever it actually is]  
- **Deployment:** [Render / Vercel / Railway / wherever DocAgent is live]  

## Try it locally  

```bash  
# Clone the repo  
git clone https://github.com/AY-Khalid/DocAgent.git  
cd DocAgent  

# Install dependencies  
pip install -r requirements.txt  

# Add your API keys  
cp .env.example .env    
# Edit .env and add OPENAI_API_KEY and GEMINI_API_KEY   

# Run the app  
uvicorn main:app --reload  
```

Then open `http://localhost:8000`.  

## Project status  

This is an actively developed prototype. The deployed version is suitable for demos and testing, not clinical use. I'm currently working on:  

- [ ] Retrieval over a clinical guidelines knowledge base (RAG)  
- [ ] Multi-language support (Hausa, Yoruba, Igbo, Pidgin)  
- [ ] Clinician-facing mode with structured note export  
- [ ] Audit logging for safety review  

## Important: not medical advice  

DocAgent is a research and education tool. It is not a medical device, has not been clinically validated, and should not be used to make medical decisions without consulting a qualified healthcare professional. If you or someone else may be in a medical emergency, contact your local emergency services immediately.  

## License  

MIT — see `LICENSE` file.  

## Get in touch  

If you're working on clinical AI in low-resource settings or want to collaborate, reach out:  

- 📧 aniduyakubu@gmail.com  
- 🐦 [@aykhalid_1](https://twitter.com/aykhalid_1)  
- 🌐 [aykhalid-portfolio.netlify.app](https://aykhalid-portfolio.netlify.app)  
