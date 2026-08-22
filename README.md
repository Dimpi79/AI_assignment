AI Engineer Intern Assignment

 Approach :

I built a small RAG based question answering system using the given logistics handbook documents. First, I load all the markdown files from the corpus and split them into smaller chunks. I used TF-IDF with cosine similarity to retrieve the top relevant chunks for a question. Only these retrieved chunks are passed to the LLM to generate the answer, and the source filename is returned as the citation. If no relevant information is found, the system returns that the handbook does not cover the question instead of making up an answer.

Tech Used

* Python
* scikit-learn (TF-IDF and cosine similarity)
* OpenAI API
* python-dotenv

 How to Run

Install the required packages:

pip install -r requirements.txt
Create a `.env` file and add the API key:
OPENAI_API_KEY=your_api_key_here
Then run:
python main.py

Retrieval

For every question, the system retrieves the top 3 relevant chunks from the handbook. I also printed the retrieval scores for a few questions to manually check whether the relevant document was being retrieved.

Task 4 - Testing and Results

I ran all 8 sample questions provided in `questions.json`.
Overall result: 8/8

The unanswerable question about the employee vacation policy was also correctly refused because this information was not present in the provided handbook.

Example Output

Question: What is the DIM divisor for international shipments?

Answer: The international DIM divisor is 166.

Citation:`dimensional-weight.md`

Supported: True

Question:What is Meridian's employee vacation policy?

Answer:I don't know. The handbook does not cover this.

Citation: None

Supported: False

Limitations

TF-IDF works well when the question uses words that are also present in the documents, but it may miss some semantically similar questions with different wording. With more time, I would try a hybrid retrieval approach using embeddings along with keyword search and add a reranking step.
