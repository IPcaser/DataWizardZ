import os
import pandas as pd
import gradio as gr
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from crewai_tools import PDFSearchTool, FileReadTool, DOCXSearchTool, CSVSearchTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# API keys-----------------move them to ENV 
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["GOOGLE_API_KEY"] = "Your API KEY"

# Load The Gemini model for LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    verbose=True,               
    temperature=0.6,            # high temp=high accuracy and low creativity                                   
    google_api_key="Your API KEY"
)

#<-----------------------------Tools----------------------------------->
class tools:
    def pdfRead(path):
        PDFtool = PDFSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                        
                    ),
                ),
            ),
            pdf=path
        )
        return PDFtool
    
    def fileRead(path):
        Filetool = FileReadTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                        
                    ),
                ),
            ),
            file_path=path
        )
        return Filetool
    
    def docsRead(path):
        Docstool = DOCXSearchTool(
            config=dict(
                llm=dict(
                    provider="google",
                    config=dict(
                        model="gemini-1.5-flash-latest",
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/msmarco-distilbert-base-v4"
                        
                    ),
                ),
            ),
            docx=path
        )
        return Docstool
#<-----------------------------Tools----------------------------------->

#<------------------------------Agents START------------------------->

class AgentLoader:

    def csvReaderAgent(path):
        agent = create_csv_agent(
            llm,
            path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        return agent

    def fileReaderAgent(path):
        FileReader = Agent(
            role='File searcher',
            goal='To analyse and generate optimal and reliable results',
            backstory="""You are a File specialist and can handle multiple file formats like .txt, .csv, .json etc.
            You are responsible to analyse the file to find the relevant content that solves the problem of the user and generate high quality and reliable results.
            You should also provide the results of your analysis and searching.""",
            llm=llm,
            verbose=True,
            tools=[tools.fileRead(path)],
            allow_delegation=False
        )
        return FileReader
    
    def PdfReaderAgent(path):
        PdfReader = Agent(
            role='PDF searcher',
            goal='To analyse and generate optimal and reliable results',
            backstory="""You are a PDF specialist and content writer.
            You are responsible to analyse the pdf to find the relevant content that solves the problem of the user and generate high quality and reliable results.
            You should also provide the results of your analysis and searching.""",
            llm=llm,
            verbose=True,
            tools=[tools.pdfRead(path)],
            allow_delegation=False,
            max_iter=6
        )
        return PdfReader
    
    def DocsReaderAgent(path):
        DocsReader = Agent(
            role='Docs searcher',
            goal='To analyse and generate optimal and reliable results',
            backstory="""You are a Docs specialist and content writer.
            You are responsible to analyse the pdf to find the relevant content that solves the problem of the user and generate high quality and reliable results.
            You should also provide the results of your analysis and searching.""",
            llm=llm,
            verbose=True,
            tools=[tools.docsRead(path)],
            allow_delegation=False
        )
        return DocsReader
    
    def writerAgent():
        writer=Agent(
            role='Content Writer',
            goal='To produce higly accurate and easy to understand information',
            backstory="""You are an content specialist and are respinsible to generate reliable and easy to understand content or information based on the summary of data.
            You should provide indetail results on the summary data.""",
            verbose=True,
            llm=llm,
            max_iter=6
        )
        return writer

#<------------------------------Agents END------------------------->

#<-------------------------------Tasks---------------------------->
def getTasks(query, agent, exp):
    task_read=Task(
        description=f'{query}',
        agent=agent,
        expected_output=f'A detailed information on {query}'
    )

    task_write=Task(
        description=f'{query}',
        agent=AgentLoader.writerAgent(),
        expected_output=exp
    )

    return [task_read, task_write]

# Gradio interface function
def process_file(file, query, expected_output):
    path = file.name
    
    if path.endswith(".pdf"):
        agent = AgentLoader.PdfReaderAgent(path)
    elif path.endswith(".docx"):
        agent = AgentLoader.DocsReaderAgent(path)
    elif path.endswith(".json") or path.endswith(".txt"):
        agent = AgentLoader.fileReaderAgent(path)
    elif path.endswith(".csv"):
        agent = AgentLoader.csvReaderAgent(path)
        results = agent.run(query)
    else:
        return 'File NOT supported'
    
    if not path.endswith(".csv"):
        task1 = getTasks(query, agent, expected_output)
        mycrew = Crew(
            agents=[agent, AgentLoader.writerAgent()],
            tasks=task1,
            verbose=True
        )
        results = mycrew.kickoff()
    
    return results

# Create the Gradio interface
interface = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Upload File"),
        gr.Textbox(label="Query"),
        gr.Textbox(label="Expected Output")
    ],
    outputs="text",
    title="File Analyzer",
    description="Upload a file (CSV, PDF, DOCX, TXT, JSON) and enter your query to get detailed information."
)

# Launch the Gradio interface
interface.launch()