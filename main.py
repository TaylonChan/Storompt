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

# Debug Libs
import langchain
import time

# langchain.debug = True

# FAMPE : A Composite Affective Model Based on Favorability, Attitude, Mood, Personality and Emotion

MODEL_PATH = "./src/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_K_M.bin"

char_human = "Boy"
char_aibot = json.load(open('./charactor/lila_nightshade.json'))

suggestion = json.dumps("")

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

{conrecord}

Your Output:
"""

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
    conrecord_prompt = PromptTemplate.from_template(
    """
Current conversation:
{history} {input}"""
    )

    template = """
I want you to act as the following charactor in first person narrative with emotion and action. Please make a verbose response to extend the plot.
{charactor}

{conrecord}
Boy: 

Your Output:
"""

    final_prompt = PromptTemplate.from_template(template)
    input_prompts = [
        ("charactor", charactor_prompt),
        ("conrecord", conrecord_prompt)
    ]
    pipeline_prompt = PipelinePromptTemplate(
        final_prompt=final_prompt, 
        pipeline_prompts=input_prompts,
        input_variables=["history", "input"]
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
        temperature=0.85,
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
prompt = create_prompt()
prompt_suggest = create_suggestion_prompt()

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

def make_suggestion(message, history):
    memory = ConversationBufferMemory(ai_prefix=char_aibot["name"], human_prefix=char_human)
    for user_msg, ai_msg in history:
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)

    suggestion_result = ConversationChain(
        prompt=prompt_suggest,
        llm=llm, 
        memory=memory,
        verbose=True
    )

    response = suggestion_result.predict(input=message)

    suggestion = json.dumps(response)
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
            ).then(
                make_suggestion, 
                [message, chatbot], 
                com_suggestion
            )

app.launch(
    ## share=True,
    inbrowser=True,
    show_error=True
)