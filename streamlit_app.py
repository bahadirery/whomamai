import streamlit as st
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from app_utils import generate_answer, get_uuid_list_from_files
import re
import streamlit.components.v1 as components
import random
import time
import re
import json

import uuid

# Assuming your function is defined in another file, import it here
# from your_script import your_function

# Placeholder for your actual function

# Setup your embeddings, database, and LLM here
embeddings_model_name = 'all-MiniLM-L12-v2'
persist_directory = 'chroma_kamy/'

model_type = "mistralai/Mistral-7B-Instruct-v0.2"


style = """
    <style>
        .text-border {
            border: 2px solid #4CAF50;  # Customize the border color and size
            padding: 10px;  # Customize the padding around the text
            font-size: 20px;  # Adjust the font size
            font-family: Arial, sans-serif;  # Choose your font
            white-space: pre-wrap;  # Preserve whitespace
        }
    </style>
    """

def filter_text(text):
    """
    Filters out lines containing UUIDs or a specific sentence pattern from the given text.

    Parameters:
    - text (str): The text to be filtered.

    Returns:
    - str: The filtered text.
    """
    # Regular expression to match a UUID
    uuid_regex = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'

    # Regular expression to match the specific line pattern
    specific_line_regex = r'This is the flashcard information for the individual with UUID [0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'

    # Filter out lines containing UUIDs or the specific sentence
    filtered_lines = [line for line in text.split('\n') if not re.search(uuid_regex, line) 
                      and not re.search(specific_line_regex, line) 
                      and not line.strip().lower().startswith("note:")
                      and not line.strip().lower().startswith("the flash card")]

    # Join the filtered lines back into a single string
    filtered_text = '\n'.join(filtered_lines)

    return filtered_text
def search_for_uuid_then_return_name(text, replace_uuid_with_name=False):
    # Regular expression to match UUIDs
    uuid_regex = r'[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'
    
    # Load the JSON file with UUID to name mappings
    with open('name_uuid_mapping.json', 'r') as file:
        uuid_to_name = json.load(file)
    
    # Reverse the dictionary to map from UUID to name
    name_to_uuid = {v: k for k, v in uuid_to_name.items()}
    
    # Function to replace each UUID found in the text with its corresponding name
    def replace_uuid(match):
        uuid = match.group(0).lower()
        return name_to_uuid.get(uuid, uuid)  # Return the name if found, else return the UUID itself

    if replace_uuid_with_name:
        # Replace all UUIDs in the text with names
        modified_text = re.sub(uuid_regex, replace_uuid, text, flags=re.IGNORECASE)
        return modified_text
    else:
        # Search for UUIDs in the given text
        found_uuids = re.findall(uuid_regex, text, re.IGNORECASE)
        # For each found UUID, find the corresponding name
        names = [name_to_uuid[uuid.lower()] for uuid in found_uuids if uuid.lower() in name_to_uuid]
        # Return the names associated with the UUIDs
        return names
    
    

# Initialize your components (embeddings, db, llm) here
@st.cache_resource
def load_embedding_model(embeddings_model_name):
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_model_name, 
        cache_folder="/local/work/baheryilmaz/.cache",
        model_kwargs={"device": 0}
        )
    return embeddings

def generate_flashcard():
    if st.session_state.uuid_list:
        selected_uuid = random.choice(st.session_state.uuid_list)
        # Assuming you have a function to generate a flashcard for the selected UUID
        flashcard = f"Flashcard for {selected_uuid}"  # Placeholder for the actual generation logic
        st.session_state.selected_uuid = selected_uuid  # Store the selected UUID in session state
        # Here you would call your actual generation logic instead of displaying the message
        st.write(flashcard)
    else:
        st.error("No UUIDs left to select!")

@st.cache_resource
def load_llm(model_type):
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_type,
        task="text-generation",
        model_kwargs={"temperature": 0.7, 
                      "do_sample": True, 
                      #"load_in_8bit": True,
                      "max_length": 10000, 
                      "trust_remote_code": True, 
                      "cache_dir": "/local/work/baheryilmaz/.cache"},
        device=0
    )
    return llm

def getRandomGIF():
    # todo random generated,
    gifs_list = ['https://media1.tenor.com/m/VVrTk5ABuiYAAAAC/mr-bean-mr.gif',
                 'https://media1.tenor.com/m/wHJyJC5427gAAAAC/waiting-waiting-patiently.gif',
                 'https://media1.tenor.com/m/_hUq1BSUsiMAAAAC/cat-cute.gif',
                 'https://media.tenor.com/qqcZIjr_LdAAAAAM/vibing-cat.gif',
                 'https://media1.tenor.com/m/MkhCAVsnwE4AAAAC/avatar-aang.gif',
                 'https://media1.tenor.com/m/1_2bArjnn1QAAAAC/patiently-waiting-patience.gif',
                 'https://media1.tenor.com/m/yztEJTm309UAAAAd/still-waiting-for-reply-few-hours-late.gif',]
    random_gif = random.choice(gifs_list)
    print(random_gif)
    return random_gif


# Streamlit app starts here
def main():
    
    if "gif_displayed" not in st.session_state:
        st.session_state.gif_displayed = False
    
    embedding_model = load_embedding_model(embeddings_model_name)
    llm = load_llm(model_type)
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)

    #st.title("WisPerMed Party WHO AM I? Guessing Game")
    #st.balloons()
    #st.snow()
    #st.success('This is a success message!', icon="✅")
    #st.toast('Mikel Bahn was the correct name')
    #st.spinner(text="In progress...")
    # also loading bar


    # Sidebar for app instructions and model information
    st.sidebar.header("Instructions")
    st.sidebar.write("To start click 'GENERATE PERSON'. Generation takes max 30 seconds.")
    st.sidebar.header("About the Model")
    st.sidebar.write("This application powered by Mistral 7b model from MistralAI. Generation happens with additional retrieval step to get personal data. This step combines the power of large language models with external knowledge sources to provide informative and contextually relevant answers.")
    st.sidebar.image('theme.png', width=300)
    # Generate a list of UUIDs for selection
    directory_path = 'source_data/individual_texts'
    
    if 'uuid_list' not in st.session_state:
        st.session_state.uuid_list = get_uuid_list_from_files(directory_path)
    
    if 'current_section_index' not in st.session_state:
        st.session_state.current_section_index = 0
    
    if 'displayed_text' not in st.session_state:
        st.session_state.displayed_text = ""
        
    if 'text_container' not in st.session_state:
        st.session_state.text_container = ""
    
    if 'name' not in st.session_state:
        st.session_state.name = ""

    if 'scoreboard' not in st.session_state:
        st.session_state.scoreboard = []

    if 'score_person' not in st.session_state:
        st.session_state.score_person = ""
    
    # Button in the sidebar that toggles the visibility of the text input
    if st.sidebar.button('Do not click', use_container_width=True):
        st.session_state['show_input'] = True  # Use session state to remember to show the input field
    
    st.session_state['reveal'] = False

    if st.sidebar.button('Reveal Person', use_container_width=True):
        st.session_state['reveal'] = True  # Use session state to remember to show the input field
    
    
    #uuid_selection = st.selectbox("Select an UUID:", st.session_state.uuid_list)

    # Main section for user interaction
    #st.header("Talk to the Chatbot")
    #st.markdown(btn_style, unsafe_allow_html=True)
    
    # Create two columns: col1 for generation, col2 for future content
    col1, col2 = st.columns([2, 1])  # Adjust the ratio between the columns as needed

    with col1:
        if st.button("GENERATE PERSON"):
            print("Generating Person...")
            st.session_state.gif_displayed = False
            if not st.session_state.get('gif_displayed', False):
                with st.spinner(text="Generating..."):


                    # Reset
                    gif_placeholder = st.empty()
                    st.session_state.text_container = st.empty()
                    st.session_state.displayed_text = ""
                    st.session_state.current_section_index = 0
                    st.session_state.sections = []


                    gif_placeholder.image(getRandomGIF(), width=500)  # Assume getRandomGIF() is defined elsewhere
                    selected_uuid = random.choice(st.session_state.get('uuid_list', []))
                  
                    #sections_completion_bar.empty()
                    final_query = f"{selected_uuid}"
                    #final_query ='3663b017-4c67-4022-93c7-43d02edb882e' # demonstration query
                    response = generate_answer(final_query, llm, db, embedding_model)

                    print("Response:!!!!!!!!!!!!!!!!!")
                    print(response)
                    
                    st.session_state.name = search_for_uuid_then_return_name(selected_uuid) 

                    response = filter_text(response)

                    response.replace("```", "")
                    
                    print("\033[91m{}\033[0m".format(st.session_state.name))
                   
                    #response = response.split('FINAL ANSWER:')[1]

                    print(response)
                    raw_sections = re.split(r'\n(?=- )', response)
                    
                    # add line break after every split
                    raw_sections = [section + "\n" for section in raw_sections]
                    
                    # remove any line that does not start with "-"
                    raw_sections = [section for section in raw_sections if section.startswith("-")]
                    
                    st.session_state.sections = raw_sections
                    
                    # randomly shuffle the sections
                    
                    random.shuffle(raw_sections)
                    
                    gif_placeholder.empty()  # Clears the GIF

                    # Remove the selected UUID to prevent re-selection
                    if 'uuid_list' in st.session_state:
                        st.session_state.uuid_list.remove(selected_uuid)
                    
                    st.session_state.gif_displayed = True
        
        # Next section button
        if 'sections' in st.session_state and st.sidebar.button("Next Section", use_container_width=True):
            if st.session_state.current_section_index < len(st.session_state.sections) - 1:
                st.session_state.current_section_index += 1
                progress_text = "Progress: {}/{} sections".format(st.session_state.current_section_index + 1, len(st.session_state.sections))
                progress_percentage = (st.session_state.current_section_index + 1) / len(st.session_state.sections)

                sections_completion_bar = st.progress(0)
                sections_completion_bar.progress(progress_percentage, text=progress_text)
            else:
                st.warning("You've reached the end of the sections.")

        

        # Display the current section
        if 'sections' in st.session_state and st.session_state.current_section_index >= 0:
            
            print('st.session_state.current_section_index')
            current_section = st.session_state.sections[st.session_state.current_section_index]
            # here a function that takes the current_section as text then streams it like it was being generated. 
            
            print(current_section)
            #st.write(current_section)
            
            text = current_section
            total_length = len(text)
            #progress_bar = st.progress(0)
            st.session_state.text_container = st.empty()
            
            # Stream the text character by character
            #Iterate through the new section character by character

            if st.session_state['reveal'] == False:
                for i, char in enumerate(text):
                    st.session_state.displayed_text += char  # Append the current character to the stored displayed text
                    
                    # Update the text container with the new cumulative content
                    #text_container.markdown(style + f"<div class='text-border'>{st.session_state.displayed_text}</div>", unsafe_allow_html=True)
                    
                    if "```" in st.session_state.displayed_text:
                        st.session_state.displayed_text.replace("```", "")
                    st.session_state.text_container.markdown(f"\n{st.session_state.displayed_text}")


                    # Update the progress bar
                    #progress_percentage = (i + 1) / total_length
                    #progress_bar.progress(progress_percentage)
                    
                    # Control the speed of text streaming (adjust as necessary)
                    time.sleep(0.02)
                
            # After displaying the full text, remove the progress bar
            #progress_bar.empty()
            # Check session state to decide whether to show the text input field

        if st.session_state['reveal'] == True:
            st.session_state['reveal'] = False
            
            #st.session_state.sections = st.session_state.sections[st.session_state.current_section_index+1:]
            st.session_state.score_person = "{}/{} sections".format(st.session_state.current_section_index + 1, len(st.session_state.sections)) # save the score
            st.session_state.current_section_index = len(st.session_state.sections)-1

            st.session_state.text_container = st.empty()
            st.session_state.displayed_text = ""
            for section in st.session_state.sections:
                st.session_state.displayed_text += section
                st.session_state.text_container.markdown(f"\n{st.session_state.displayed_text}")
            
            #st.header('The person is: ', st.session_state.name[0])
           

        if st.session_state.get('show_input', False):
            user_input = st.text_input('Enter your input here:', key='user_input_for_chatbot')  # Add a key to ensure widget uniqueness
            
            # Button to trigger processing of the input
            if st.button('Enter'):
                if user_input:  # Check if user_input is not empty
                    with st.spinner(text="Generating..."):
                        response = generate_answer(user_input, llm, db, embedding_model, chatbot=True)
                        #response = response.split('FINAL ANSWER:')[1]
                        names = search_for_uuid_then_return_name(response)
                        st.write('Output:', response)

                        for names in names:

                            st.write('Individuals:\n', names)
                        st.session_state['show_input'] = False  # Optionally hide the input field after processing
                else:
                    st.write('Please enter some input.')

    st.sidebar.header(f"Participants left: {len(st.session_state.uuid_list)}")
    
    with col2:
        if 'sections' in st.session_state and st.session_state.current_section_index == len(st.session_state.sections)-1 :
            # Placeholder for future content
            st.session_state.scoreboard.append([st.session_state.name[0],st.session_state.score_person]) 
        
        if len(st.session_state.scoreboard) > 0:
            st.header("Scoreboard")

            for name in st.session_state.scoreboard:
                st.write(f"{name[0]} - {name[1]}")
                
        
             
            
if __name__ == "__main__":
    st.set_page_config(
        page_title="Who_am_AI_chatbot",
        page_icon="🤖",
        layout="wide",
    )
    main()