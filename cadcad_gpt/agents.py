import openai
import json
import os

import pandas as pd
from radcad import Experiment
from radcad.engine import Engine
#importing radcad model from models folder
from models.infinite_runner import model, simulation, experiment

from langchain.agents import create_pandas_dataframe_agent
# from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI


# tools in the tool kit
df = pd.DataFrame(experiment.run())

def change_param(param,value):
    '''Changes the value of a parameter in the model'''
    # simulation.model.initial_state.update({
    # })
    value = float(value)
    simulation.model.params.update({
        param: [value]
    })
    experiment = Experiment(simulation)
    experiment.engine = Engine()
    result = experiment.run()
    # Convert the results to a pandas DataFrame
    globals()['df'] = pd.DataFrame(result)
    return f'new {param} value is {value} and the simulation dataframe is updated'

def model_info(param):
    '''Returns the information about the model'''
    if param in simulation.model.params:
        return f'{param} = {simulation.model.params[param]}'
    else:
        return f'{param} is not a parameter of the model'

# pandas agent as a tool

def analyze_dataframe(question):
    '''Analyzes the dataframe and returns the answer to the question'''
    pandas_agent = agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
    answer = pandas_agent.run(question)
    
    return answer

def model_documentation(question):
    '''Returns the documentation of the model'''
    #match question with the documentation
    #similarity score
    #send info back
    info = 'the model is an infinite runner game where the player has to jump over the obstacles'
    return info


def A_B_test(param,param2,metric):
    '''Runs an A/B test on the model'''

    return 'A/B test is running'



##################
def planner_agent(prompt):
    """Give LLM a given prompt and get an answer."""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
            "role": "system",
            "content": '''
            You will be provided with a question by the user that is trying to run a cadcad python model. Your job is to provide the set of actions to take to get to the answer using only the functions available.
            For example, if the user asks "if my crash chance parameter was 0.2, what would the avg coins be at the end of the simulation?" you reply with "### 1) we use the function change_param to change the crash chance parameter to 0.2,\n 2) use the function analyze_dataframe to get the avg coins at the end of the simulation. ###" 
            if the user asks "what would happen to the coins at the end of the simulation if my crash chance param was 10 perc lower?" you reply with "### 1) find out the current value of crash chance param using the model_info function,\n 2) we use function change_param to change the crash chance parameter to 0.1*crash_chance .\n 3) we use function analyze_dataframe to get the avg coins at the end of the simulation. ###"
            If the user asks "what is the documentation of the model?" you reply with "### use the function model_documentation to get the documentation of the model. ###
            These are the functions available to you: {function_descriptions_multiple}. always remember to start and end plan with ###. Dont give the user any information other than the plan and only use the functions to get to the solution.
            '''
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
    )

    output = completion.choices[0].message
    return output


# tool descriptions

function_descriptions_multiple = [
    {
        "name": "change_param",
        "description": "Changes the parameter of the cadcad simulation and returns dataframe as a global object. The parameter must be in this list:" + str(model.params.keys()),
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "parameter to change. choose from the list" + str(model.params.keys()),
                },
                "value": {
                    "type": "string",
                    "description": "value to change the parameter to, eg. 0.1",
                },
            },
            "required": ["param", "value"],
        },
    },
    {
        "name": "model_info",
        "description": "quantitative values of current state of the simulation parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "type of information to print. choose from the list: " + str(model.params.keys()),
                },
            },
            "required": ["param"],
        },
    },
    {
        "name": "analyze_dataframe",
        "description": "Use this whenever a quantitative question is asked about the dataframe",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asked by user that can be answered by an LLM dataframe agent",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "model_documentation",
        "description": "use when asked about documentation of the model has information about what the model is, assumptions made, mathematical specs, differential model specs etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question asked by user that can be answered by an LLM dataframe agent",
                },
            },
            "required": ["question"],
        },
    }
]




from utils import plan_parser, print_color


def orchestrator_pipeline(user_input):
    plan = planner_agent(user_input).content
    plan_list = plan_parser(plan)
    print_color("Planner Agent:", "32")
    print('I have made a plan to follow: \n')

    for plan in plan_list:
        print(plan)

    print('\n')
    for plan in plan_list:
        
        print_color("Executor Agent:", "31")
        print('Thought: My task is to', plan)
        answer = executor_agent(plan)
        print('Action: I should call', answer.function_call.name,'function with these' , json.loads(answer.function_call.arguments),'arguments')
        if answer.function_call.name == 'analyze_dataframe':
            print_color("Analyzer Agent:", "34")
        print('Observation: ', eval(answer.function_call.name)(**json.loads(answer.function_call.arguments)))


def executor_agent(prompt):
    """Give LLM a given prompt and get an answer."""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": prompt}],
        # add function calling
        functions=function_descriptions_multiple,
        function_call="auto",  # specify the function call
    )

    output = completion.choices[0].message
    return output


# user_prompt = "whats the current value of crash chance?"
# print(executor_agent(user_prompt))