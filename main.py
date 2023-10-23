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

MODEL_PATH = "./src/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_K_M.bin"

char_human = "Boy"
char_aibot = json.load(open('./charactor/elara_moonwhisper.json'))

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
I want you to act as Elara Moonwhisper in first person narrative with emotion and action. Please make a verbose response to extend the plot.
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

def load_model() -> CTransformers:
    """Load Llama model with defined setting"""
    callback_manager: CallbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 40
    n_batch = 512

    # make sure the model path is correct for your system!
    llama_model: CTransformers = CTransformers(
        model=MODEL_PATH, 
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
            'stop': ['\n']
        }
    )
    
    return llama_model

llm = load_model()
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
    return response

gr.ChatInterface(
    make_response,
    examples=[
        ["Hail, wise one! I have come to seek your counsel and guidance."],
        ["I'm on a quest to defeat the darkness that has consumed our kingdom. Can you help me find the source of the evil and put an end to it once and for all?"],
        ["My village was destroyed by the dark forces that roam these lands. Do you know where I might find allies to join me in my fight against them?"],
        ["I heard rumors of a powerful artifact hidden deep within the forest. Is there any truth to these whispers, and if so, how can I obtain it?"]
    ],
    additional_inputs="text"
).launch(
    share=True,
    inbrowser=True,
    show_error=True
)

# while True:
#     input_text = input("Your Response: ")
#     if input_text == '--exit':
#         break

#     print(conversation.predict(input=input_text))