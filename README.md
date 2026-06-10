# Researcher: an ai assistant for research

this project demostrates how to create an ai assitant to do reasoning heavy tasks with agentic RAG of local and fetched resources.

this is the brach based on langgraph.

## dev notes

I currently implement it in following way:

process local resource into txt or markdown using [GLM-OCR]

feed the processed content to the RAG system in langgraph,
with embedding model as [Qwen text embedding].

configure langgraph to use agentic RAG system with logical reasoning skills.

# current dev status:

./code
- main.ipynb #main code for implementation
- ./data
--/local #used for local resources
--/fetched #used for fetched resources from the web
./reasoner
prompts for the reasoner agent
./reference
some reference materials for the implementation, including docs and code references.



