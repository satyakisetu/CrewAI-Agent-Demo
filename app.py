from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from dotenv import load_dotenv


load_dotenv()

topic = "Medical Industry using GenAI"

#Tool 1
llm = LLM(model="gpt-4")

#Tool 2
search_tool = SerperDevTool(n=10)

#Agent 1
senior_research_analyst = Agent(
    role="Senior Research Analyst",
    goal="Conduct thorough research on {topic}",
    backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=llm
)

#Agent 2

content_writer = Agent(
    role="Content Writer",
    goal="Transform research findings into a comprehensive article on {topic} while maintaining accuracy and readability",
    backstory="You are a skilled writer with experience in creating \
               engaging and informative content in medical field avoiding plagiarism",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

#Task 1: Research Tasks

reseach_tasks = Task(
    description=f"A detailed summary of the research findings, including key points, recent development and news, key industry trends and innovations,\
         expert opinions and advice, statistical data and market insights, evaluate source credibility and fact checks and include all relevant citations\
             and sources about the topic: {topic}",
    agent=senior_research_analyst,
    expected_output="A detailed summary of the research findings, including key points, recent development and news, key industry trends and innovations,\
         expert opinions and advice, statistical data and market insights, evaluate source credibility and fact checks and include all relevant citations\
             and sources about the topic {topic}"
    
)

#Task 2: Writing Task

writing_task = Task(
    description=f"Write a comprehensive article on the topic: {topic} based on the research findings provided by the Senior Research Analyst",
    agent=content_writer,
    expected_output="A comprehensive article on the topic {topic} based on the research findings provided by the Senior Research Analyst"
)
 

crew = Crew(agents=[senior_research_analyst, content_writer], 
             tasks=[reseach_tasks, writing_task],
             verbose=True)

result = crew.kickoff(input={"topic": topic})