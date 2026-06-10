An agent is a model calling tools in a loop until a given task is complete.
```
                                                Agent = Model + Harness
```

<div align = "center" >
  <img width="300" height="268" alt="image" src="https://github.com/user-attachments/assets/4a6a3ea3-bf51-43c4-90eb-623ba2c3a38b"/>
</div>

**Harness** : A harness is everything around that loop: the model, its prompt, its tools, and any middleware that shapes its behavior.The job of a harness to get the model the right context at the right time for the given task.
### create_agent
create_agent is a highly configurable harness. At its simplest:
```python
from langchain.agents import create_agent

agent = create_agent("openai:gpt-5.4", tools=tools)
```
Configure the basics directly — model=, tools=, system_prompt=. For more advanced capabilities, extend the harness using middleware.
