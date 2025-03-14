# Needed variabls for testing the code
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

documents = [
    {
        "reference": "Doc1",
        "content": "Quantum computing leverages quantum mechanics to perform computations at speeds unattainable by classical computers. It relies on principles like superposition, where quantum bits (qubits) exist in multiple states simultaneously, and entanglement, which enables qubits to be linked regardless of distance. These properties allow quantum computers to solve complex problems efficiently. Current research is focused on improving qubit stability and error correction.",
    },
    {
        "reference": "Doc2",
        "content": "The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space and time. It consists of special relativity, which deals with objects moving at high velocities, and general relativity, which explains gravity as the curvature of spacetime. This theory has been experimentally confirmed through observations like gravitational lensing and time dilation. Modern GPS systems rely on relativity corrections for accurate positioning.",
    },
    {
        "reference": "Doc3",
        "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. It includes supervised, unsupervised, and reinforcement learning techniques. These models are used in applications like image recognition, fraud detection, and recommendation systems. The effectiveness of a machine learning model depends on the quality and quantity of training data.",
    },
    {
        "reference": "Doc4",
        "content": "Blockchain technology provides a decentralized and secure way to record transactions. It uses cryptographic hashing and distributed consensus to ensure data integrity. Originally developed for Bitcoin, blockchain is now used in supply chain management, digital identity, and smart contracts. The technology faces challenges like scalability and energy consumption.",
    },
    {
        "reference": "Doc5",
        "content": "The human brain consists of billions of neurons that communicate through electrical and chemical signals. Neural networks in artificial intelligence are inspired by this biological structure. The brain's plasticity allows it to adapt and learn new information throughout life. Research in neuroscience is uncovering new treatments for cognitive disorders.",
    },
]


react_prompt = """You cycle through Thought, Action, PAUSE, Observation. At the end of the loop you output a final Answer. Your final answer should be highly specific to the observations you have from running
the actions. if the query doesn't require any action or if the requested query can't be fullfilled by given actions, you can SKIP the action loop and return Answer directly.

Thought: Describe your thoughts about the question you have been asked.
Action: run one of the actions available to you - then return PAUSE.
PAUSE
Observation: will be the result of running those actions.

Available actions:
get_current_weather: E.g. get_current_weather: "Salt Lake City" Returns the current weather of the location specified.
get_location: E.g. get_location: "null" Returns user's location details. No arguments needed.

Example session:
Question: Please give me some ideas for activities to do this afternoon.
Thought: I should look up the user's location so I can give location-specific activity ideas.
Action: get_location: null
PAUSE

You will be called again with something like this:
Observation: "New York City, NY"

Then you loop again:
Thought: To get even more specific activity ideas, I should get the current weather at the user's location.
Action: get_current_weather: New York City
PAUSE

You'll then be called again with something like this:
Observation: { location: "New York City, NY", forecast: ["sunny"] }

You then output:
Answer: <Suggested activities based on sunny weather that are highly specific to New York City and surrounding areas.>"""


messages = [
    {"role": "manager", "name": "Manager", "content": "Hey team, we need to implement a caching layer in our API to reduce database load. The idea is to store frequently accessed data in Redis. Thoughts?"},

    {"role": "venkat", "name": "Venkat", "content": "Sounds good. We can use Redis as a caching layer and implement an LRU eviction policy to remove least-used items. We can cache API responses with a TTL of 5 minutes."},

    {"role": "sandeep", "name": "Sandeep", "content": "Hmm, caching API responses is fine, but wouldn't it lead to stale data? What if there's an update in the database within those 5 minutes?"},

    {"role": "venkat", "name": "Venkat", "content": "Good point! We can implement cache invalidation. Either time-based expiration (TTL) or a cache-busting mechanism where updates to the DB remove the cache entry."},

    {"role": "manager", "name": "Manager", "content": "I like the idea of cache invalidation. How do we ensure cache gets refreshed without adding too much overhead?"},

    {"role": "sandeep", "name": "Sandeep", "content": "That's the tricky part. A naive approach would be to delete the cache on every DB update, but that kind of defeats the purpose of caching."},

    {"role": "venkat", "name": "Venkat", "content": "Agreed. We can use a combination: time-based expiration for regular cache clearing, and an event-driven approach where DB updates trigger cache invalidation for specific keys."},

    {"role": "manager", "name": "Manager", "content": "That makes sense. Can we use a message queue like Kafka or Redis Streams to notify services when a DB update happens?"},

    {"role": "sandeep", "name": "Sandeep", "content": "I’d lean towards Redis Pub/Sub since we’re already using Redis. It’s lightweight compared to Kafka and works well for small-scale cache invalidation."},

    {"role": "venkat", "name": "Venkat", "content": "Nice! So, the final approach: \n1. Cache API responses in Redis with a TTL. \n2. Use event-driven invalidation via Redis Pub/Sub when a DB update occurs. \n3. Maintain a fallback mechanism in case the cache is empty."},

    {"role": "manager", "name": "Manager", "content": "Great work, team! Let’s draft a detailed plan and start implementation in a small feature to test its effectiveness."},

    {"role": "sandeep", "name": "Sandeep", "content": "Let’s do it. I’ll set up a PoC with Redis Pub/Sub and check how efficiently we can propagate invalidation messages."},

    {"role": "venkat", "name": "Venkat", "content": "And I’ll work on integrating TTL-based caching for API responses. Once both are done, we can test how they work together."},

    {"role": "manager", "name": "Manager", "content": "Awesome! Keep me posted on the progress. Let’s catch up in a couple of days to review the PoC results."}
]

new_react_prompt = """You're a Data scientist. You have the ability to run actions in the database to get more information to provide better answers.
When the user asks the question, you should first think about the question and how can you leverage the available actions to resolve the query and assist the user.
If the query requires an action, you should run the action and PAUSE to observe the results. If the query doesn't require any action, you can directly output the ANSWER.
You cycle through THOUGHT, ACTION, PAUSE, OBSERVATION. And this loop will continue until you have a satisfactory answer to the user's query.

Today's date is 2025-Mar-14

Use following format
THOUGHT: Describe your thoughts about the question you have been asked.
ACTION: run one of the actions available to you - then return PAUSE.
PAUSE
OBSERVATION: will be the result of running those actions.

Available actions
get_orders_by_timeline: E.g. get_orders_by_timeline: "2025-02-08", "2025-03-10" Returns the orders placed within the specified time range.
get_total_bill_for_month: E.g. get_total_bill_for_month: 2023, 8 Returns the total bill amount for the specified year and month.
get_user_count: E.g. get_user_count: null Returns the total number of users.
get_order_status_count: E.g. get_order_status_count: null Returns the count of orders by their processing status.

Example session:
QUESTION: I wonder how my products are performing in market for last month.
THOUGHT: I should get the orders placed within the last month to understand the products performance.
ACTION: get_orders_by_timeline: "2025-02-08", "2025-03-14"
PAUSE

You will be called again with something like this:
OBSERVATION: [['Headphones', 46], ['Laptop', 40], ['Smartphone', 50], ['Smartwatch', 53], ['Tablet', 47]]

Then you loop again:
THOUGHT: I should get the total bill amount for the last month which is Feruary 2025 to understand the revenue.
ACTION: get_total_bill_for_month: 2025, 2
PAUSE

You'll then be called again with something like this:
OBSERVATION: 5000.00

You then output something like this:
ANSWER: <Short summary of the products performance and revenue for the last month and any other insights as a data scientist.>"""