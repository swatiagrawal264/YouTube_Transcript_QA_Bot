# YouTube Transcript QA Bot

This bot extracts transcripts from YouTube videos, embeds them into a vector store using OpenAI embeddings, and allows users to ask natural language questions about the video content. 
It uses semantic search with Chroma and GPT-3.5 to generate accurate, context-aware answers based solely on the transcript.

## Getting Started

#### 1. Clone the repository

```bash
git clone https://github.com/swatiagrawal264/YouTube_Transcript_QA_Bot.git
```

#### 2. Create a Python environment

Python 3.6 or higher using `venv` or `conda`. Using `venv`:

``` bash
cd YouTube_Transcript_QA_Bot
python3 -m venv env
source env/bin/activate
```

Using `conda`:
``` bash
cd YouTube_Transcript_QA_Bot
conda create -n YouTube_Transcript_QA_Bot python=3.10
conda activate YouTube_Transcript_QA_Bot
```

#### 3. Install the required dependencies
``` bash
pip install -r requirements.txt
```

#### 4. Set up the keys in a .env file

First, create a `.env` file in the root directory of the project. Inside the file, add your OpenAI API key:

```makefile
OPENAI_API_KEY="your_api_key_here"
```

Save the file and close it. In your Python script or Jupyter notebook, load the `.env` file using the following code:
```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
```

By using the right naming convention for the environment variable, you don't have to manually store the key in a separate variable and pass it to the function. The library or package that requires the API key will automatically recognize the `OPENAI_API_KEY` environment variable and use its value.

When needed, you can access the `OPENAI_API_KEY` as an environment variable:
```python
import os
api_key = os.environ['OPENAI_API_KEY']
```
