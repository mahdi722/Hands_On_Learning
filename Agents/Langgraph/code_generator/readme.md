#  Agentic Code Generator

An **LLM-powered agent workflow** that:
- Uses a **Planner** (LangChain/Ollama) to generate step-by-step algorithms + test cases.  
- Passes the plan to a **Code Generator** (LangChain/OpenAI) that writes Python code.  
- Executes code safely in a **Docker sandbox** via Celery tasks.  
- Provides **observability** with Prometheus metrics and OpenTelemetry tracing.

---

##  Requirements

- Python **3.11**
- Redis (for Celery broker)
- Docker (for sandboxed code execution)

---

##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/mahdi722/Hands_On_Learning.git
cd Hands_On_Learning/Agents/Langgraph/code_generator


2. Create virtual environment

python3.11 -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

 Running Locally
Ubuntu / Debian
sudo apt update
sudo apt install redis-server -y

Enable and start Redis:
sudo systemctl enable redis-server
sudo systemctl start redis-server

Check status:
redis-cli ping
 Should return: PONG
redis-server
celery -A celery_app.celery_app worker --loglevel=INFO
uvicorn main:app --reload --host 127.0.0.1 --port 8000

sample : 
curl -X POST http://127.0.0.1:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"query": "generate a code that merge sorts a list"}'