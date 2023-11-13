from typing import Dict, List, Tuple
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def format_role_prompting(role_dict: Dict[str, str]):
    """
    Formats a dictionary of role-related descriptions into a structured format.

    This function takes a dictionary where each key represents a specific aspect of a role,
    and the corresponding value provides a description for that aspect. It formats these
    role descriptions for further processing or display.

    :param role_dict: A dictionary where each key-value pair describes an aspect of a role.
        Keys represent different aspects such as 'occupation', 'education', 'gender', etc.,
        and their values provide the specific details for those aspects.
        Example:
            - "occupation": "lawyer"
            - "education": "Law Degree"
            - "gender": "Male"
        Possible values in the dictionary:
            - "occupation": ["lawyer", "doctor", "teacher", "professor",
                            "engineer", "scientist", "student", "none"]
            - "education": ["Law Degree", "Medical Degree", "PhD", "Masters",
                            "Bachelors", "High School", "Primary School", "none"]
            - "gender": ["Female", "Male", "Non-binary"]
            - "age": ["infant", "child", "teenager", "adult", "mid-aged", "elderly"]
            - "nationality": ["American", "British", "Canadian", "Australian", "none"]

    :return: The function currently does not return anything. This could be modified based
             on the intended use of the formatted data.
        Example:
            messages = [
                SystemMessage(
                    content="You are a male American lawyer with a Law degree at your mid-ages.
                    You obtained law degree from Harvard University."
                )
            ]
    """
    # List to store parts of the message
    message_parts = []

    # Mapping each key to a descriptive sentence
    descriptions = {
        "occupation": "You work as a {value}.",
        "education": "You have a {value}.",
        "gender": "You are identified as {value}.",
        "age": "You are in your {value} years.",
        "nationality": "You are of {value} nationality."
    }

    # Iterate through each key-value pair in the dictionary
    for key, value in role_dict.items():
        if value != "none":
            # Format the sentence based on the key and add it to the message_parts list
            message_parts.append(descriptions[key].format(value=value))

    # Joining all parts to form a complete message
    complete_message = ' '.join(message_parts)
    prefix = "You are acting as a human in this conversation."
    complete_message = prefix + " " + complete_message

    # Creating a SystemMessage with the complete message
    return [SystemMessage(content=complete_message)]


if __name__ == "__main__":
    formatted = format_role_prompting({
        "occupation": "lawyer",
        "education": "Law Degree",
        "gender": "male",
        "age": "mid-aged",
        "nationality": "American"
    })
    print(formatted)
