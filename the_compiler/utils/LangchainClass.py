#provide access to alot of tools, meta prompt, add worker agent, add boss agent, load balancing auto scaling for a workload
from langchain.experimental.autonomous_agents.autogpt.agent import AutoGPT
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.docstore.document import Document
import asyncio
import nest_asyncio
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import faiss
from langchain.vectorstores import FAISS
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun
from abc import ABC, abstractmethod
import os
from contextlib import contextmanager
from typing import Optional
from langchain.agents import tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool

from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import Field
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain, BaseCombineDocumentsChain
import playwright

def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
    )
# !pip install playwright
# !playwright install
async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results

def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)

@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))

class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = "Browse a webpage and retrieve the information relevant to the question."
    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=_get_text_splitter)
    qa_chain: BaseCombineDocumentsChain
    
    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 4):
            input_docs = web_docs[i:i+4]
            window_result = self.qa_chain({"input_documents": input_docs, "question": question}, return_only_outputs=True)
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [Document(page_content="\n".join(results), metadata={"source": url})]
        return self.qa_chain({"input_documents": results_docs, "question": question}, return_only_outputs=True)
    
    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError
      

ROOT_DIR = "./data/"

@contextmanager
def pushd(new_dir):
    """Context manager for changing the current working directory."""
    prev_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev_dir)

@tool
def process_csv(
    csv_file_path: str, instructions: str, output_path: Optional[str] = None
) -> str:
    """Process a CSV by with pandas in a limited REPL.\
 Only use this after writing data to disk as a csv file.\
 Any figures must be saved to disk to be viewed by the human.\
 Instructions should be written in natural language, not code. Assume the dataframe is already loaded."""
    with pushd(ROOT_DIR):
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            return f"Error: {e}"
        agent = create_pandas_dataframe_agent(llm, df, max_iterations=30, verbose=True)
        if output_path is not None:
            instructions += f" Save output to disk at {output_path}"
        try:
            result = agent.run(instructions)
            return result
        except Exception as e:
            return f"Error: {e}"

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass



embeddings_model = OpenAIEmbeddings()
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

from langchain.agents import tool
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.write import WriteFileTool

# !pip install duckduckgo_search
web_search = DuckDuckGoSearchRun()
query_website_tool = WebpageQATool(qa_chain=load_qa_with_sources_chain(llm))


class CustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model
        tools = [
            web_search,
            WriteFileTool(root_dir="./data"),
            ReadFileTool(root_dir="./data"),
            process_csv,
            query_website_tool,
            # HumanInputRun(), # Activate if you want the permit asking for help from the human
        ],
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name="Tom",
            ai_role="Assistant",
            tools=tools,
            llm=model,
            memory=vectorstore.as_retriever(search_kwargs={"k": 8}),
        )

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        prompt = f"Given the current state of reasoning: '{state_text}', generate {k} coherent thoughts to continue the reasoning process:"
        response = self.agent.arun(input=prompt)
        thoughts = response.strip().split('\n')
        return thoughts

    def evaluate_states(self, states):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the following states of reasoning, vote for the best state:\n{state_text}\n\nVote, and NOTHING ELSE:"
            response = self.agent.arun(input=prompt)
            try:
                value = float(response)
                print(f"value: {value}")
            except ValueError:
                value = 0  # Assign a default value if the conversion fails
            state_values[state] = value
        return state_values

    def main(self, task, max_iters=3, max_meta_iters=5):
        failed_phrase = 'task failed'
        success_phrase = 'task succeeded'
        key_phrases = [success_phrase, failed_phrase]

        instructions = 'None'
        for i in range(max_meta_iters):
            print(f'[Episode {i+1}/{max_meta_iters}]')
            chain = initialize_chain(instructions, memory=None)
            output = chain.predict(human_input=task)
            for j in range(max_iters):
                print(f'(Step {j+1}/{max_iters})')
                print(f'Assistant: {output}')
                print(f'Human: ')
                human_input = input()
                if any(phrase in human_input.lower() for phrase in key_phrases):
                    break
                output = chain.predict(human_input=human_input)
            if success_phrase in human_input.lower():
                print(f'You succeeded! Thanks for playing!')
                return
            meta_chain = initialize_meta_chain()
            meta_output = meta_chain.predict(chat_history=get_chat_history(chain.memory))
            print(f'Feedback: {meta_output}')
            instructions = get_new_instructions(meta_output)
            print(f'New Instructions: {instructions}')
            print('\n'+'#'*80+'\n')
        print(f'You failed! Thanks for playing!')