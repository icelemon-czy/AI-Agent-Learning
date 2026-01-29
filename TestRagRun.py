import os
import time
from Tools.buildin.rag import Rag

def test_rag_api():
    print("=== Testing Rag API Actions ===")
    
    # 1. Initialize Rag
    print("\n1. Initializing Rag...")
    try:
        rag = Rag(knowledgeBaseURL="./test_kb_temp", collection="test_collection", namespace="test_case")
        if not rag.initialized:
            print("Failed to initialize RAG.")
            return
        print("Rag initialized success.")
    except Exception as e:
        print(f"Error initializing Rag: {e}")
        return

    # Create dummy file
    dummy_file = "test_doc_1.txt"
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write("# Machine Learning\nMachine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.")
    
    dummy_file_2 = "test_doc_2.txt"
    with open(dummy_file_2, "w", encoding="utf-8") as f:
        f.write("# Python\nPython is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.")

    try:
        # 2. Test 'add_document'
        print("\n2. Testing 'add_document'...")
        res = rag.run({
            "action": "add_document",
            "file_path": dummy_file
        })
        print(f"Result:\n{res}")

        # 3. Test 'add_text'
        print("\n3. Testing 'add_text'...")
        res = rag.run({
            "action": "add_text",
            "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
            "document_id": "manual_text_1"
        })
        print(f"Result:\n{res}")

        # 4. Test 'stats'
        print("\n4. Testing 'stats'...")
        res = rag.run({
            "action": "stats",
            "namespace": "test_case"
        })
        print(f"Result:\n{res}")

        # 5. Test 'search'
        print("\n5. Testing 'search'...")
        res = rag.run({
            "action": "search",
            "query": "learning algorithms",
            "top_k": 2
        })
        print(f"Result:\n{res}")

        # 6. Test 'ask'
        print("\n6. Testing 'ask'...")
        # Note: This might make an LLM call. If LLM is not configured, it might fail or return error.
        res = rag.run({
            "action": "ask",
            "question": "What is machine learning?",
            "limit": 3
        })
        print(f"Result:\n{res}")
        
    finally:
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        if os.path.exists(dummy_file_2):
            os.remove(dummy_file_2)
        print("\n=== Test Finished ===")

if __name__ == "__main__":
    test_rag_api()
