# WIP

docker-compose up -d

docker exec -it postgres_db bash
 docker exec -it postgres_db psql -U myuser -d myappdb

cleaning
docker-compose down -v
docker-compose up -d


cd common
uvicorn ai_core.main:app --reload

streamlit run rag_ui.py
http://localhost:8000/docs

curl -X 'POST' \
  'http://localhost:8000/api/llm/generate_text' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "What is the meaning of life?",
  "max_new_tokens": 10,
  "skip_special_tokens": true,
  "kwargs": {
  "temperature": 0.9
}
}'

curl -X 'POST' \
  'http://localhost:8000/api/llm/generate_response' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain relativity in simple terms.",
  "system_prompt": "You are a science teacher.",
  "kwargs": {}
}'



curl -X 'POST' \
  'http://localhost:8000/api/llm/stateless_chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Can you write a short poem for me please",
  "chat_history": [
    {
      "content": "Hi",
      "role": "user"
    },
    {
      "content": "Hello! What can I do for you?",
      "role": "assistant"
    }
  ],
  "kwargs": {}
}'

curl -X 'POST' \
  'http://localhost:8000/api/llm/brainstorm' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
   {"role": "Deepak", "content": "Team, we have a new feature request from Britney. Let'\''s discuss the feasibility and timeline."},

    {"role": "Britney", "content": "Yes, I want to add an **advanced search feature** to our platform. Users are struggling to find relevant items quickly."},

    {"role": "Venkat", "content": "That sounds interesting. Britney, do you have any specific filters in mind? Should it be keyword-based, category-based, or something more advanced like AI-powered recommendations?"},

    {"role": "Britney", "content": "A combination of both! Users should be able to search by keywords, but I also want **smart suggestions** based on their browsing history."},

    {"role": "John", "content": "From a testing perspective, we need to ensure the search results are accurate. Venkat, how complex will the AI recommendations be? We might need test cases for various user behaviors."}

  ],
  "role": "",
  "iam": "Britney",
  "role_play_configs": [
    {"role": "Deepak", "persona": "A meticulous and strategic manager who ensures projects run smoothly. He focuses on business goals, deadlines, and resource allocation."},
    {"role": "Britney", "persona": "A business client who values user experience and is focused on solving real-world problems for customers."},
    {"role": "Venkat", "persona": "A skilled developer with deep technical expertise. He prioritizes efficiency, clean code, and optimal system design."},
    {"role": "John", "persona": "A detail-oriented tester who thrives on finding edge cases and ensuring product stability."}
  ],
  "ai_assisted_turns": 4,
  "kwargs": {}
}'