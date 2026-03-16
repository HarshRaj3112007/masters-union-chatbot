# 🎓 Masters Union DSAI Course Assistant

An **AI-powered conversational assistant** for the Masters Union Undergraduate Programme in Data Science & Artificial Intelligence. Provides instant answers to student queries about curriculum, fees, admissions, career outcomes, campus facilities, and more.

---

## 🌟 Features

- **Intelligent Q&A**: Asks questions about curriculum, admissions, fees, career outcomes, global immersion programs, faculty, and campus life
- **Streaming Responses**: Real-time answer generation with typing indicators for better UX
- **Hybrid Retrieval**: Combines vector embeddings (semantic) + BM25 (keyword-based) for accurate answers
- **Comparison Mode**: Side-by-side comparison of programme aspects (e.g., "Compare Year 1 vs Year 2")
- **Confidence Badges**: Displays confidence levels (high/medium/low) for answer reliability
- **Source Citations**: View the exact source material used to answer your question
- **Query Classification**: Automatically categorizes questions (fees, curriculum, career, etc.)
- **Follow-up Suggestions**: Dynamic suggestions for related questions based on context
- **Query History**: Sidebar with clickable past questions for easy navigation
- **Suggested Questions**: Pre-loaded questions to get started quickly

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                           │
│                    (app.py)                                     │
│  - Chat interface                                               │
│  - Suggested questions                                          │
│  - Query history sidebar                                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Backend (naya.py)                        │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Ingestion                                          │  │
│  │  - DSAI_Brochure_Structured.txt                          │  │
│  │  - scraped_web_data.txt (from scraper.py)               │  │
│  │  - program_data.txt                                      │  │
│  │  - additionaldata.txt                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Chunking & Embedding                                    │  │
│  │  - RecursiveCharacterTextSplitter                        │  │
│  │  - Sentence Transformers (all-MiniLM-L6-v2)             │  │
│  │  - ChromaDB (Vector Storage)                             │  │
│  │  - BM25 (Keyword Index)                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Query Processing                                         │  │
│  │  - Query Expansion (HyDE)                                │  │
│  │  - Hybrid Retrieval (Semantic + Keyword)                 │  │
│  │  - Reciprocal Rank Fusion (RRF) Re-ranking              │  │
│  │  - Topic Classification                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM Generation                                           │  │
│  │  - Groq LLM (Llama 3.1 8B Instant)                       │  │
│  │  - Streaming responses                                   │  │
│  │  - Confidence scoring                                    │  │
│  │  - Follow-up generation                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Groq API Key**: Free tier available at [console.groq.com](https://console.groq.com)
- **pip**: Python package manager
- **macOS/Linux/Windows**: Tested on all platforms

---

## ⚙️ Setup Instructions

### 1. **Clone or Download the Project**

```bash
cd /Users/harshraj/Desktop/Xpecto
```

### 2. **Create a Virtual Environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

**Dependencies included:**

- `streamlit>=1.30.0` - Web framework for UI
- `chromadb>=0.4.22` - Vector database for embeddings
- `sentence-transformers>=2.2.2` - Embedding model
- `langchain-text-splitters>=0.0.1` - Text chunking
- `groq>=0.4.0` - LLM API client
- `requests>=2.31.0` - HTTP client for web scraping
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=5.0.0` - XML/HTML processing
- `rank-bm25>=0.2.2` - BM25 keyword search
- `python-dotenv>=1.0.0` - Environment variable management

### 4. **Set Up Environment Variables**

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

**Get a Groq API Key:**

1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Generate an API key
4. Paste it in the `.env` file

### 5. **Prepare Data Files** (Optional)

The application uses these data sources:

- `DSAI_Brochure_Structured.txt` - Official brochure content
- `scraped_web_data.txt` - Content scraped from Masters Union website
- `program_data.txt` - Additional programme information
- `additionaldata.txt` - Supplementary data

To update web data, run the scraper:

```bash
python scraper.py
```

---

## 🚀 Running the Application

### Start the Streamlit Server

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501` (or another port if 8501 is in use).

### Access the Assistant

1. Open the URL shown in the terminal
2. See the list of **Suggested Questions** - click any to start
3. Or type your own question in the input field
4. View chat history in the sidebar
5. Click **View source context** to see the RAG retrieval results

---

## 📖 Usage Guide

### **Asking Questions**

**Suggested Topics:**

- 💰 **Fees & Scholarships** - Fee structure, payment plans, scholarships
- 📚 **Curriculum** - Year-wise subjects, course details, learning outcomes
- ✅ **Eligibility & Admissions** - JEE/SAT requirements, application process
- 💼 **Career Outcomes** - Placements, salary ranges, recruiter companies
- 🌍 **Global Immersion** - Illinois Tech degree, international internships
- 🏫 **Campus Life** - Facilities, hostels, clubs, location
- 🧑‍🏫 **Faculty & Mentors** - Industry experts, guest speakers
- 📝 **How to Apply** - Step-by-step admissions process

### **Query Examples**

```
❓ "What is the complete fee structure?"
❓ "How many years is the programme?"
❓ "Compare Year 1 and Year 2 curriculum"
❓ "What big tech companies hire from this programme?"
❓ "Am I eligible if I don't have a JEE score?"
❓ "Tell me about the global immersion track"
❓ "What is the placement success rate?"
```

### **Understanding Confidence Levels**

- 🟢 **High Confidence** - Answer fully supported by source materials
- 🟡 **Medium Confidence** - Partial information or inference applied
- 🔴 **Low Confidence** - Limited source data, may need verification

---

## 🔧 Project Files & Their Roles

| File                           | Purpose                                           |
| ------------------------------ | ------------------------------------------------- |
| `app.py`                       | Streamlit UI - main entry point                   |
| `naya.py`                      | RAG backend - query processing & LLM integration  |
| `scraper.py`                   | Web scraper - updates programme data from website |
| `requirements.txt`             | Python dependencies                               |
| `.env`                         | API keys (create manually, add to .gitignore)     |
| `DSAI_Brochure_Structured.txt` | Official programme brochure                       |
| `scraped_web_data.txt`         | Live website content (generated by scraper.py)    |
| `program_data.txt`             | Structured programme metadata                     |
| `additionaldata.txt`           | Supplementary information                         |

---

## 💡 Core Components Explained

### **1. Query Classification** (`naya.py`)

Automatically categorizes questions into topics:

- Fees | Curriculum | Admissions | Career | Global | Campus | Faculty | Contact

### **2. Hybrid Retrieval**

- **Semantic Search**: Uses sentence embeddings to find contextually similar passages
- **BM25 Search**: Keyword-based matching for exact term matching
- **RRF Re-ranking**: Reciprocal Rank Fusion combines both results

### **3. LLM Generation** (Groq API)

- Uses **Llama 3.1 8B** for fast, cost-effective responses
- Generates confidence scores based on source relevance
- Creates follow-up question suggestions
- Streams responses for better UX

### **4. Comparison Mode**

Detects comparison queries like "Compare X vs Y" and structures side-by-side answers.

---

## 🎯 Use Cases

### **For Prospective Students**

- ✅ Quick answers to admission questions
- ✅ Understand fee structure and payment options
- ✅ Explore curriculum before applying
- ✅ Learn about career outcomes and placements

### **For Current Students**

- ✅ Navigate curriculum across years
- ✅ Understand global immersion opportunities
- ✅ Get campus life information
- ✅ Connect with faculty and mentors

### **For Parents**

- ✅ Fee and scholarship details
- ✅ Campus facilities and safety
- ✅ Career outcomes and ROI
- ✅ Contact information for queries

### **For Administrators**

- ✅ Reduce repetitive admissions queries
- ✅ Provide 24/7 information access
- ✅ Maintain consistent messaging
- ✅ Track frequently asked questions

---

## 🔄 Updating Programme Information

### **Option 1: Update Web Content**

```bash
python scraper.py
```

This refreshes `scraped_web_data.txt` from the Masters Union website.

### **Option 2: Manual Data Updates**

Edit these files directly:

- `program_data.txt` - Core programme facts
- `additionaldata.txt` - Recent updates, news, etc.

The RAG system will automatically re-index on next restart.

---

## 🛠️ Troubleshooting

### **Error: GROQ_API_KEY not set**

```
Solution: Ensure .env file exists in project root with GROQ_API_KEY=<your_key>
```

### **Error: Module not found**

```
Solution: pip install -r requirements.txt
```

### **Streamlit not starting**

```bash
# Kill existing process
lsof -i :8501
kill -9 <PID>

# Restart
streamlit run app.py
```

### **Slow responses**

- Check internet connection (Groq API requires connectivity)
- Vector embeddings are computed once and cached by ChromaDB
- First query may be slower due to initialization

### **Low confidence answers**

- Update data files with more specific information
- Run `scraper.py` to refresh web content
- Add relevant FAQs to `additionaldata.txt`

---

## 📊 Performance & Optimization

**Current Configuration:**

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim, ~22MB)
- **LLM**: Llama 3.1 8B Instant (fast, low latency)
- **Chunk Size**: 500 tokens with 50-token overlap
- **Top-K Retrieval**: 5 documents from vector DB + 5 from BM25
- **Token Limit**: ~2000 tokens per response

**Tips for Optimization:**

- Use a dedicated machine for production deployment
- Consider increasing chunk size for domain-specific content
- Cache frequently asked questions separately
- Use [Groq's free tier](https://console.groq.com) for development

---

## 🔐 Security Considerations

- ✅ **API Key**: Keep GROQ_API_KEY in `.env` (add to `.gitignore`)
- ✅ **No Data Logging**: The assistant doesn't store user conversations
- ✅ **Groq Privacy**: Check [Groq's privacy policy](https://groq.com/privacy/)
- ✅ **Production Deployment**: Use environment-specific `.env` files

---

## 📞 Support & Contact

For issues or questions:

1. Check the **Troubleshooting** section above
2. Review **Suggested Questions** in the app for common topics
3. Run `python scraper.py` to refresh data
4. Contact Masters Union admissions at **ugadmissions@mastersunion.org**

---

## 📜 License

This project is built for the Masters Union Undergraduate Data Science & AI Programme. All programme content belongs to Masters Union.

---

## 🚀 Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Create `.env` with Groq API key
3. ✅ Run the app: `streamlit run app.py`
4. ✅ Ask your first question!
5. ✅ (Optional) Update data: `python scraper.py`

**Happy learning!** 🎓✨
