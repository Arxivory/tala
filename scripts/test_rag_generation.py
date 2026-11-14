
from tala.rag.generation import generate_response
from tala.rag.schemas import RAGRequest
from dotenv import load_dotenv

load_dotenv()

questions = [
    "How does microgravity affect muscle strength, and why is exercise important for astronauts?",
    "How can fatigue and sleep loss affect human performance?",
    "How do gravity and physical capabilities influence human biomechanics?",
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:", question)
    print("=" * 80)

    request = RAGRequest(
        question=question,
        top_k=4,
    )

    result = generate_response(request)

    print("\nANSWER")
    print(result.answer)

    print("\nEVIDENCE")
    for item in result.evidence:
        metadata = item.metadata
        print(
            f"- {metadata.title} "
            f"(pages {metadata.page_start}-{metadata.page_end})"
        )

    print("\nLIMITATIONS")
    for limitation in result.limitations:
        print("-", limitation)

    print("\nABSTAINED:", result.abstained)

    if result.abstained:
        print("REASON:", result.abstention_reason)
