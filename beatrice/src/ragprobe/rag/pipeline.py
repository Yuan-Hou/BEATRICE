class RAGPipeline:
    """Minimal placeholder for future real-RAG experiments.

    Retriever, generator, and probe filter can all be attached here.
    The current project stabilizes the oracle-context RAG and probe flow first.
    """

    def __init__(self, retriever=None, generator=None, probe=None):
        self.retriever = retriever
        self.generator = generator
        self.probe = probe

    def answer(self, question: str, top_k: int = 5):
        if self.retriever is None or self.generator is None:
            raise RuntimeError("RAGPipeline requires both retriever and generator.")
        contexts = self.retriever.retrieve(question, top_k=top_k)
        return self.generator.generate(question, contexts)
