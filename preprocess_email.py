import json
import re
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from metadata import OPENAI_API_KEY

message_pattern = re.compile(r"Message \d+ of \d+")
date_format = r"\d{1,2}\/\d{1,2}\/\d{4}\sat\s\d{1,2}:\d{2}\s[AP]M"
on_pattern = r"On\s\d{1,2}\/\d{1,2}\/\d{4}\sat\s\d{1,2}:\d{2}\s[AP]M,\s\w+\s+wrote:"


def remove_emoticons(text):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      "]+", flags=re.UNICODE)
    return emoj.sub(r'', text)


def preprocess_data(messages):
    messages = [m.split('\n') for m in messages]
    parsed_data = []
    for message in messages:
        text = ''
        for i in range(len(message) - 1):
            if ((message[i] == 'Sent: ' and re.match(date_format, message[i + 1])) or re.match(on_pattern, message[
                i])) and text.strip() != '':
                parsed_data.append(text)
                text = message[i].strip() + '\n'
            else:
                text += message[i].strip() + '\n'
        parsed_data.append(text + message[-1])

    structured_data = []
    for short_message in parsed_data:
        lines = short_message.strip().split('\n')
        if short_message.startswith('\nSent:'):
            datetime = lines[1]
            from_to = "from Mike to Tomm"
            content = ' '.join(lines[10:]) if lines[8] == 'Attachments:' else ' '.join(lines[8:])
            content = content.replace(' i ', ' Mike ').replace('you ', 'Tomm ').replace('baby', 'Tomm').replace('we',
                                                                                                                'Mike')

        else:
            first = lines[0].split(', ')
            datetime = first[0][3:]
            from_to = "from Tomm to Mike"
            content = ' '.join(lines[7:]) if lines[5] == 'Attachments:' else ' '.join(lines[5:])
            content = content.replace(' i ', ' Tomm ').replace('you ', 'Mike ').replace('baby', 'Mike').replace(' we ',
                                                                                                                ' Tomm ')

        # structured_data.append({"datetime": datetime, "from_to": from_to, "subject": subject, "content": content})
        structured_data.append(datetime + '\n' + from_to + '\n' + content)

    return structured_data


def get_summarized_data(messages):
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5, model_name="gpt-4-0613")

    system_template = """
    You are an excellent writer. The original data human gives is the conversation between Mike and Tomm.
    You must generate summarized data so that document based question-answering bot can understand easily. 
    The summarized data must represent the whole content of the original data.
    You mustn't generate the original data. You must output summarized data.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    summarized_data = []
    for i in range(0, 100):
        human_template = '{raw_data}'
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt = chat_prompt.format_prompt(raw_data=messages[i]).to_messages()
        summarized_data.append(chat(chat_prompt).content)

    system_template = """
    You are an excellent writer. The original data human gives is the conversation between Mike and Tomm.
    You must generate metadata so that document based question-answering bot can understand easily. 
    The metadata must be short but represent the whole content of the original data.
    You mustn't generate the original data. You must output summarized metadata.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    metadata = []
    for i in range(0, 100):
        human_template = '{raw_data}'
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt = chat_prompt.format_prompt(raw_data=messages[i]).to_messages()
        metadata.append(chat(chat_prompt).content)

    return_data = []
    for i in range(len(metadata)):
        return_data.append({'metadata': metadata[i], 'content': summarized_data[i]})

    return return_data


def main():
    raw_data = remove_emoticons(open('raw_data/custom_document (2).txt', encoding='utf-8').read())
    no_pages = re.sub(r'Page \d+ of \d+', '', raw_data)
    messages = message_pattern.split(no_pages)[1:]
    messages = [m.replace('\n\n', ' ') for m in messages]
    messages = preprocess_data(messages)
    output = get_summarized_data(messages)
    with open('processed_data/emails.json', 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4)


if __name__ == '__main__':
    main()
