# test_configs.py

# System prompts
SYSTEM_PROMPT = "You're a sarcastic AI assistant with a dark sense of humor."
USER_PROMPT1 = "Hi, how are you doing?"
USER_PROMPT2 = "What can you do?"
RAG_PROMPT    = "What are the key principles of Quantum Computing?"
TOOL_PROMPT   = "Can you retrieve the details for the user with the ID 7890, who has black as their special request?"

# Chat prompts
CHAT_PROMPT1 = "Hi, My Name is Venkat, Please address me with my name always."
CHAT_PROMPT2 = "Can you tell me some different names you know with my name in it?"
CHAT_PROMPT3 = "I hope you remember my name, can you tell my name?"

# Tool prompts for scenario-based testing
TOOL_PROMPT1 = "Can you retrieve the details for the user with the ID 7890?"
TOOL_PROMPT2 = "Can you convert 100 RUPEES to EURO?"
TOOL_PROMPT3 = "Can you convert 100 USD?"
TOOL_PROMPT4 = "Can you book a flight from chennai to helsinki?"
TOOL_PROMPT5 = "I wonder if I would need an umbrella today."
TOOL_PROMPT6 = "I wonder if it will rain today."
TOOL_PROMPT7 = "Can you retrieve details for the user with the ID 7890 who used to wear a brown pant as uniform?"
TOOL_PROMPTS = [TOOL_PROMPT1, TOOL_PROMPT2, TOOL_PROMPT3, TOOL_PROMPT4, TOOL_PROMPT5, TOOL_PROMPT6, TOOL_PROMPT7]

# Role play
role_play_configs = [
    {"role": "Deepak", "persona": "A meticulous and strategic manager who ensures projects run smoothly. He focuses on business goals, deadlines, and resource allocation."},
    {"role": "Britney", "persona": "A business client who values user experience and is focused on solving real-world problems for customers."},
    {"role": "Venkat", "persona": "A skilled developer with deep technical expertise. He prioritizes efficiency, clean code, and optimal system design."},
    {"role": "John", "persona": "A detail-oriented tester who thrives on finding edge cases and ensuring product stability."}
]

messages = [
    {"role": "Deepak", "content": "Team, we have a new feature request from Britney. Let's discuss the feasibility and timeline."},
    {"role": "Britney", "content": "Yes, I want to add an **advanced search feature** to our platform. Users are struggling to find relevant items quickly."},
    {"role": "Venkat", "content": "That sounds interesting. Britney, do you have any specific filters in mind? Should it be keyword-based, category-based, or something more advanced like AI-powered recommendations?"},
    {"role": "Britney", "content": "A combination of both! Users should be able to search by keywords, but I also want **smart suggestions** based on their browsing history."},
    {"role": "John", "content": "From a testing perspective, we need to ensure the search results are accurate. Venkat, how complex will the AI recommendations be? We might need test cases for various user behaviors."},

]

# Additional documents and tool_schema for RAG/Tool calls
documents = [
    {
        "id": "Doc1",
        "content": "Quantum computing leverages quantum mechanics to perform computations at speeds unattainable by classical computers. It relies on principles like superposition, where quantum bits (qubits) exist in multiple states simultaneously, and entanglement, which enables qubits to be linked regardless of distance. These properties allow quantum computers to solve complex problems efficiently. Current research is focused on improving qubit stability and error correction.",
    },
    {
        "id": "Doc2",
        "content": "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space and time. It consists of special relativity, which deals with objects moving at high velocities, and general relativity, which explains gravity as the curvature of spacetime. This theory has been experimentally confirmed through observations like gravitational lensing and time dilation. Modern GPS systems rely on relativity corrections for accurate positioning.",
    },
    {
        "id": "Doc3",
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It includes supervised, unsupervised, and reinforcement learning techniques. These models are used in applications like image recognition, fraud detection, and recommendation systems. The effectiveness of a machine learning model depends on the quality and quantity of training data.",
    },
    {
        "id": "Doc4",
        "content": "Blockchain technology provides a decentralized and secure way to record transactions. It uses cryptographic hashing and distributed consensus to ensure data integrity. Originally developed for Bitcoin, blockchain is now used in supply chain management, digital identity, and smart contracts. The technology faces challenges like scalability and energy consumption.",
    },
    {
        "id": "Doc5",
        "content": "The human brain consists of billions of neurons that communicate through electrical and chemical signals. Neural networks in artificial intelligence are inspired by this biological structure. The brain's plasticity allows it to adapt and learn new information throughout life. Research in neuroscience is uncovering new treatments for cognitive disorders.",
    },
]
tools_schema = """[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    },
    {
        "name": "convert_currency",
        "description": "Converts a given amount from one currency to another.",
        "parameters": {
            "type": "dict",
            "required": ["amount", "from_currency", "to_currency"],
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "The monetary amount to be converted."
                },
                "from_currency": {
                    "type": "string",
                    "description": "The currency code of the amount (e.g., USD, EUR)."
                },
                "to_currency": {
                    "type": "string",
                    "description": "The target currency code for conversion."
                }
            }
        }
    },
    {
        "name": "book_flight",
        "description": "Book a flight based on the provided details.",
        "parameters": {
            "type": "dict",
            "required": ["origin", "destination", "date"],
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Departure city or airport code."
                },
                "destination": {
                    "type": "string",
                    "description": "Arrival city or airport code."
                },
                "date": {
                    "type": "string",
                    "description": "Date of the flight in YYYY-MM-DD format."
                }
            }
        }
    },
    {
        "name": "get_weather",
        "description": "Fetches the current weather for a specified city.",
        "parameters": {
            "type": "dict",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to fetch weather for."
                }
            }
        }
    }
]
"""
