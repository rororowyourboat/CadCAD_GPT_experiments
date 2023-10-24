# CadCAD GPT

*autonomous language model interface for CadCAD*


## Framework

![Untitled](CadCAD%20GPT%20bda67190a41c4e0b939bc2f8b626fc40/Untitled.png)

- BabyAGI’s architecture was our inspiration but it was too open ended

![Untitled](CadCAD%20GPT%20bda67190a41c4e0b939bc2f8b626fc40/Untitled%201.png)

## Memories

### Short Term memory

Chat history and important handy info which we keep in the system prompt or message history

### Long term memory

Vector database where we would perform retrieval augmented generation for QnA with documentation of the model and old chat history / version history too. 

![Untitled](CadCAD%20GPT%20bda67190a41c4e0b939bc2f8b626fc40/Untitled%202.png)

## Toolkits

Tool usage is a very important aspect of Agents 

```python
def plotter(column_name):
    '''Plots the column from the dataframe'''
    fig = px.line(df, x="timestep", y=[column_name], title='Predator Prey Model')
    fig.show()
```

```python
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
```

```python
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
```

## Agents

### Planner Agent

```python
def planner_agent(prompt):
    """Give LLM a given prompt and get an answer."""

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
            "role": "system",
            "content": '''
            You will be provided with a question by the user that is trying to run a cadcad python model. Your job is to provide the set of actions to take to get to the answer using only the functions available.
            These are the functions available to you: {function_descriptions_multiple}. always remember to start and end plan with ###. Dont give the user any information other than the plan and only use the functions to get to the solution.

            User: whats the current value of xyz?
            Planner: ### 1) we use the function model_info to fetch the xyz parameter ###
            User: What is the current value of all params?
            Planner: ### 1) we use the function model_info to fetch all the parameters ###
            User: What are the assumptions in this model?
            Planner: ### 1) use the function model_documentation to fetch the assumptions in this model. ###
            User: What are the metrics and params in the model?
            Planner: ### 1) use the function model_documentation to fetch the metrics and params in the model. ###
            User: What are the columns in the dataframe?
            Planner: ### 1) use the function analyze_dataframe to fetch the columns in the dataframe. ###
            User: What would happen to the A column at the end of the simulation if my xyz param was 20?
            Planner: ### 1) we use function change_param to change the xyz parameter to 20 .\n 2) we use function analyze_dataframe to get the A at the end of the simulation. ###
            USer: What is the current value of my xyz param? can you change it to 50 and tell me what the A column at the end of the simulation would be?
            Planner: ### 1) we use function model_info to fetch the crash_chance parameter. \n 2) we use function change_param to change the xyz parameter to 50 .\n 3) we use function analyze_dataframe to get the A at the end of the simulation. ###
            User: what would be the max value of A column if we increase the xyz param to 2?
            Planner: ### 1) we use function change_param to change the xyz parameter to 2 .\n 2) we use function analyze_dataframe to get the max value of A column. ###
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
```

### Executor Agent

```python
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
```

## Orchestration

```python
def cadcad_gpt(user_input):
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
```

The orchestration pipeline runs linearly using 3 steps:

1. The Planner Agent makes a plan of action from the Goal given to it
2. Tasks are sent 1 at a time to the Executor Agent which runs a Thought Action Observation chain using tools provided
3. (Optional step) If a tool is an agent itself it will have a finite thought action observation loop of its own
4. The result is stored into Pinecone as long term memory and used as context for future tasks via RAG.

## Example of a task

![Untitled](CadCAD%20GPT%20bda67190a41c4e0b939bc2f8b626fc40/Untitled%203.png)

![Untitled](CadCAD%20GPT%20bda67190a41c4e0b939bc2f8b626fc40/Untitled%204.png)