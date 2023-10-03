import json
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from metadata import OPENAI_API_KEY


def remove_emoticons(text):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      "]+", flags=re.UNICODE)
    return emoj.sub(r'', text)


def preprocess_data(message):
    lines = message.strip().split("\n")
    datetime = lines[1].split()[0:2]
    dt = " ".join(datetime)
    fromOrto = lines[1].split()[2]
    from_to = "from Mike to Tomm" if fromOrto == 'from' else "from Tomm to Mike"
    content = "\n".join(lines[2:]).strip().lower()
    if fromOrto == 'from':
        content = content.replace(' i ', ' Mike ').replace('you ', 'Tomm ').replace('baby', 'Tomm')
    else:
        content = content.replace(' i ', ' Tomm ').replace('you ', 'Mike ').replace('baby', 'Mike')
    return dt + '\n' + from_to + '\n' + content


def get_summarized_data(messages):
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5, model_name="gpt-4-0613")

    system_template = """
    You are an excellent writer. The original data human gives is the conversation between Mike and Tomm.
    You must generate summarized data so that document based question-answering bot can understand easily.
    The summarized data must show the whole content of the original data.
    You mustn't generate the original data. You must output summarized data.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    summarized_data = []
    for i in range(0, len(messages), 5):
        raw_data = "\n_________".join(messages[i: i + 5])
        human_template = '{raw_data}'
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt = chat_prompt.format_prompt(raw_data=raw_data).to_messages()
        summarized_data.append(chat(chat_prompt).content)

    system_template = """
       You are an excellent writer. The original data human gives is the conversation between Mike and Tomm.
       You must generate metadata so that document based question-answering bot can understand easily. 
       The metadata must be short but represent the whole content of the original data.
       You mustn't generate the original data. You must output summarized metadata.
       """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    metadata = []
    for i in range(0, len(messages), 5):
        raw_data = "\n_________".join(messages[i: i + 5])
        human_template = '{raw_data}'
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt = chat_prompt.format_prompt(raw_data=raw_data).to_messages()
        metadata.append(chat(chat_prompt).content)

    return_data = []
    for i in range(len(metadata)):
        return_data.append({'metadata': metadata[i], 'content': summarized_data[i]})

    return return_data


def main():
    raw_data = remove_emoticons(open('raw_data/custom_document (1).txt', encoding='utf-8').read())
    messages = raw_data.split("----------------------------------------------------")[1:]
    messages = [preprocess_data(message) for message in messages]
    # print(len(messages))
    output = get_summarized_data(messages)
    with open('processed_data/messages.json', 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4)


if __name__ == '__main__':
    main()
