def generate_answer(context, question, model):
    prompt = f"""Use ONLY the context below to answer the question.

Context: {context}
Question: {question}
Answer:"""
    response = model.generate_content(prompt)
    return response.text.strip()

def faithfulness_score(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())
    return len(answer_words & context_words) / max(1, len(answer_words))

def answer_relevance_score(answer, query):
    answer_words = set(answer.lower().split())
    query_words = set(query.lower().split())
    return len(answer_words & query_words) / max(1, len(query_words))

def rag_pipeline(query, retriever, model, k=3):
    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in retrieved_docs[:k]]
    )
    answer = generate_answer(context, query, model)

    return {
        "query": query,
        "answer": answer,
        "faithfulness": faithfulness_score(answer, context),
        "relevance": answer_relevance_score(answer, query),
    }
