import gradio as gr
import json

traits = ['Adventurous','Ambitious','Analytical','Assertive','Calm','Cheerful','Confident','Creative','Curious','Determined','Diligent','Dynamic','Energetic','Exuberant','Flexible','Foolhardy','Generous','Humorous','Intuitive','Joyful','Kind','Logical','Optimistic','Outgoing','Pragmatic','Reliable','Sensitive','Spontaneous','Tenacious','Thoughtful','Witty']

def make_charactor(name, age, gender, personality, template, examples):
    charactor = {
        "_type": "prompt",
        "input_variables": [],
        "name": name,
        "age": age,
        "gender": gender,
        "personality": personality,
        "template": template,
        "examples": examples
    }
    return charactor

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            input = [
                gr.Textbox(label="Name"), 
                gr.Slider(
                    0, 
                    100, 
                    value=20, 
                    label="Age", 
                    info="Age of the charactor"
                ),
                gr.Radio(
                    ["Male", "Female", "Other", "Unknown"], 
                    label="Gender", 
                ),
                gr.CheckboxGroup(
                    traits, 
                    label="Personality", 
                    info="Charactors you want the charactor contains"
                ),
                gr.Textbox(label="Background", info="Description of the character"), 
                gr.Textbox(label="Examples", info="First message to send to users")
            ]

        with gr.Column():
            output = gr.JSON()

    btn = gr.Button("Generate")
    btn.click(
        make_charactor, 
        input,
        output
    )

    gr.Examples(
        [
            [
                "Elara Moonwhisper", 
                25, 
                "Female",
                ['Calm', 'Reliable', 'Generous'],
                "Elara Moonwhisper is a 25-year-old commoner with long silver hair and emerald green eyes. She is a shy and reserved young woman who is highly skilled in the art of healing and navigating the forests and wilderness of the kingdom. Her goal is to bring peace and prosperity to the land and prove herself as more than just a simple commoner. She is fiercely independent and values her freedom above all else, and her primary motivation is to help others. Elara's greatest fear is failure and being trapped and powerless.",
                "Greetings, traveler! *gives a warm smile* My name is Elara Moonwhisper, and I am a skilled healer and navigator of the forests and wilderness of this kingdom. *looks around* I have been watching you from afar, and I sense that you are on a quest for something. *eyebrows furrow* What is it that you seek, traveler?"
            ]
        ],
        input
    )

if __name__ == "__main__":
    app.launch(
        share=True,
        show_tips=True
    )