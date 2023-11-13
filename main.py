import langchain

from langchain.llms import CTransformers
from langchain import LLMChain
from langchain.prompts import PromptTemplate, load_prompt
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.chat import (
    SystemMessage, 
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import gradio as gr
import json

# langchain.debug = True

# FAMPE : A Composite Affective Model Based on Favorability, Attitude, Mood, Personality and Emotion

MODEL_PATH = "./src/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_K_M.bin"

char_human = "Boy"
char_aibot = json.load(open('./charactor/aurora_nightshade.json'))

suggestion = [
    "Hello, wise one!",
    "Can you help me?",
    "Do you know where I might find allies to join me in my fight against them?",
]

def create_prompt() -> PipelinePromptTemplate:
    """Creates prompt template"""
    charactor_prompt = PromptTemplate.from_template(char_aibot["template"])
    conrecord_prompt = PromptTemplate.from_template(
    """
Current conversation:
{history}
{human}: {input}
{aibot}:""",
        partial_variables={
            "human": char_human,
            "aibot": char_aibot["name"]
        }
    )
    example_prompt = PromptTemplate.from_template(char_aibot["examples"])

    template = """
I want you to act as the following charactor in first person narrative with emotion and action. Please make a verbose response to extend the plot.
{charactor}

Example:
{example}

{conrecord}"""

    final_prompt = PromptTemplate.from_template(template)
    input_prompts = [
        ("charactor", charactor_prompt),
        ("example", example_prompt),
        ("conrecord", conrecord_prompt)
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=final_prompt, 
        pipeline_prompts=input_prompts, 
        input_variables=["history", "input"]
    )

    return pipeline_prompt

def create_suggestion_prompt() -> PipelinePromptTemplate:
    """Creates suggestion prompt template"""
    charactor_prompt = PromptTemplate.from_template(char_aibot["template"])

    template = """
I want you to generate 3 possible options(responses) for the following charactor in the current conversation for a role playing game with following background infomation in JSON list format.
{charactor}

Your response should be a list of comma separated values, eg:
["I don't think we should trust him. He could be working for the other side.", "Let's bring him in and see what he knows. We might need his expertise to take down our target.", "I don't like this. Let's get out of here while we still can."]

Current conversation:
Boy: Hello
Aurora Nightshade: Hi! My name is Aurora and I am 21 years old with short black hair and piercing blue eyes. You know how people say some places are haunted? Well, I can actually see those ghosts. It started when I was a kid, and now, I'm determined to make sense of it all. Do you believe in ghosts too? Maybe we can help each other figure out what's going on.
Boy: No, I don't
Aurora Nightshade: Really? That's interesting. So, have you ever experienced anything strange or unusual?
Boy: 

Your Output:
"""

    final_prompt = PromptTemplate.from_template(template)
    input_prompts = [
        ("charactor", charactor_prompt)
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=final_prompt, 
        pipeline_prompts=input_prompts
    )
    return pipeline_prompt

def load_model(model_path, stop=['\n']) -> CTransformers:
    """Load Llama model with defined setting"""
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 40
    n_batch = 512

    # make sure the model path is correct for your system!
    llama_model: CTransformers = CTransformers(
        model=model_path, 
        model_type='llama', 
        verbose=True,
        temperature=0.5,
        max_new_token=2048,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        callback_manager=callback_manager,
        config={
            'context_length': 2048,
            'stream': True,
            'stop': stop
        }
    )
    
    return llama_model

llm = load_model(MODEL_PATH)
llm_suggest = load_model(MODEL_PATH, None)
prompt = create_prompt()

def make_response(message, history, additional):
    memory = ConversationBufferMemory(ai_prefix=char_aibot["name"], human_prefix=char_human)
    for user_msg, ai_msg in history:
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)

    conversation = ConversationChain(
        prompt=prompt,
        llm=llm, 
        memory=memory,
        verbose=True
    )

    response = conversation.predict(input=message)

    history.append((message, response))
    return "", history

def make_suggestion(chatbot):

    suggestion_prompt = create_suggestion_prompt()
    result = LLMChain(
        prompt=suggestion_prompt,
        llm=llm_suggest, 
        verbose=True
    )
    suggestion = json.dumps(result.predict())
    return suggestion

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            gr.JSON(
                {
                    "favorability": 0,
                },
                label="Emotion",
                show_label=True
            )
            com_suggestion = gr.JSON(
                suggestion,
                label="Suggestion",
                show_label=True
            )
        with gr.Column():
            chatbot = gr.Chatbot(
                bubble_full_width=False,
                show_share_button=True,
                show_copy_button=True,
                likeable=True
            )
            message = gr.Textbox(
                label="Your Response"
            )
            with gr.Row():
                submit = gr.Button(
                    value="✨ Deliver",
                    scale=3
                )
                clear = gr.ClearButton(
                    [message, chatbot],
                    value="🗑️ Clear"
                )
            gr.on(
                [message.submit, submit.click],
                make_response, 
                [message, chatbot], 
                [message, chatbot]
            )

    chatbot.change(
        make_suggestion,
        inputs=chatbot,
        outputs=com_suggestion
    )

app.launch(
    ## share=True,
    inbrowser=True,
    show_error=True
)