
from tala.rag.retrieval import retrieve_evidence
from tala.rag.schemas import RAGRequest


questions = [
    "How does microgravity affect muscle strength, and why is exercise important for astronauts?",
    "How can fatigue and sleep loss affect human performance?",
    "How do gravity and physical capabilities influence human biomechanics?",
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    evidence = retrieve_evidence(
        RAGRequest(question=question, top_k=3)
    )

    if not evidence:
        print("No evidence retrieved.")
        continue

    for index, item in enumerate(evidence, start=1):
        metadata = item.metadata

        print(f"\n[{index}] {metadata.title}")
        print(f"Source ID: {metadata.source_id}")
        print(f"Pages: {metadata.page_start}-{metadata.page_end}")
        print(f"Population: {metadata.population}")
        print(f"Distance: {item.distance:.4f}")
        print("Content:")
        print(item.content[:1000])
