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

from transformers import pipeline

import gradio as gr
import json

# Debug Libs
import langchain
import time

# langchain.debug = True

# FAMPE : A Composite Affective Model Based on Favorability, Attitude, Mood, Personality and Emotion

MODEL_PATH = "./src/models/Wizard-Vicuna-13B-Uncensored.ggmlv3.q5_K_M.bin"

charactor_name = "lila_nightshade"
char_human = "Boy"
char_aibot = json.load(open(f'./charactor/{charactor_name}.json'))

suggestion = json.dumps("")

distilled_student_sentiment_classifier = pipeline(
    model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", 
    return_all_scores=True
)

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
    greeting_prompt = PromptTemplate.from_template(char_aibot["greeting"])

    template = """
I want you to act as the following charactor in first person narrative with emotion and action. Please make a verbose response to extend the plot.
{charactor}

Example:
{greeting}

{conrecord}"""

    final_prompt = PromptTemplate.from_template(template)
    input_prompts = [
        ("charactor", charactor_prompt),
        ("greeting", greeting_prompt),
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

{conrecord}
Boy:"""

    final_prompt = PromptTemplate.from_template(template)
    input_prompts = [
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

def make_response(message, history):
    memory = ConversationBufferMemory(ai_prefix=char_aibot["name"], human_prefix=char_human)
    for user_msg, ai_msg in history:
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)

    while True:
        conversation = ConversationChain(
            prompt=prompt,
            llm=llm, 
            memory=memory,
            verbose=True
        )

        response = conversation.predict(input=message)
        if not response or response.isspace():
            print("Empty Response")
        else:
            break

    history.append((message, response))
    return "", history

def make_suggestion(message, history):
    memory = ConversationBufferMemory(ai_prefix=char_aibot["name"], human_prefix=char_human)
    for user_msg, ai_msg in history:
        memory.chat_memory.add_user_message(user_msg)
        memory.chat_memory.add_ai_message(ai_msg)

    while True:
        suggestion_result = ConversationChain(
            prompt=prompt_suggest,
            llm=llm, 
            memory=memory,
            verbose=True
        )

        response = suggestion_result.predict(input=message)
        if not response or response.isspace():
            print("Empty Response")
        else:
            break

    suggestion = json.dumps(response)
    return suggestion

def classify_sentiment(chatbot):
    global favorability
    result = distilled_student_sentiment_classifier(chatbot[-1][-2])[0]
    change = 0

    positive, neutral, negative = [result[i]["score"] for i in range(3)]

    if max([positive, neutral, negative]) != neutral:
        change = positive - negative

    favorability += change
    return round(favorability, 2)

def change_charactor(charactor):
    global charactor_name, char_aibot
    charactor_name = charactor
    char_aibot = json.load(open(f'./charactor/{charactor_name}.json'))
    init_chat()

def init_chat():
    global llm, prompt, prompt_suggest, favorability
    favorability = 0
    llm = load_model(MODEL_PATH)
    prompt = create_prompt()
    prompt_suggest = create_suggestion_prompt()

init_chat()

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            radio_char = gr.Radio(
                [
                    ("Elara Moonwhisper", "elara_moonwhisper"),
                    ("Lila Nightshade", "lila_nightshade"),
                    ("Eirax Darkhunter", "eirax_darkhunter"),
                    ("Aurora Nightshade", "aurora_nightshade"),
                    ("Arcturus Starseeker", "arcturus_starseeker"),
                    ("Arcturus Ironwood", "arcturus_ironwood")
                ],
                label="Please choose a charactor",
                show_label=True
            )
            emotion = gr.Slider(
                -10,
                10,
                favorability,
                label="Emotion",
                show_label=True,
                interactive=False
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

            # Triggered when submit
            gr.on(
                [message.submit, submit.click],
                make_response, 
                [message, chatbot], 
                [message, chatbot]
            ).then(
                classify_sentiment,
                chatbot,
                emotion
            ).then(
                make_suggestion, 
                [message, chatbot], 
                com_suggestion
            )

            radio_char.change(
                change_charactor,
                radio_char,
            ).then(
                None,
                js="window.location.reload()"
            )

app.launch(
    ## share=True,
    inbrowser=True,
    show_error=True
)