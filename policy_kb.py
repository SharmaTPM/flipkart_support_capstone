import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- Task 1: Policy Knowledge Base (12 Documents) ---
POLICY_DOCUMENTS = {
    "doc_1": "Apparel and Footwear return window is 14 days from delivery date. Items must be unworn, unwashed, and have original tags intact. Refund or exchange is processed upon successful quality check during pickup.",
    "doc_2": "Electronics return window is restricted to 7 days from delivery date. Returns are accepted only for manufacturing defects or damaged items. Brand warranty applies for technical issues beyond 7 days.",
    "doc_3": "Home and Kitchen products carry a 10-day return window. Items must include all original accessories, user manuals, and packaging. Damaged items must be reported within 24 hours of delivery.",
    "doc_4": "Cash on Delivery (COD) refunds are processed directly to user verified bank accounts within 3 to 5 business days after pickup. Users must provide valid account details or UPI ID on the Flipkart portal.",
    "doc_5": "Prepaid order refunds (Credit Card, Debit Card, Net Banking, UPI) are auto-credited back to the original payment source within 24 to 48 hours following product pickup verification.",
    "doc_6": "Standard Delivery Service Level Agreement (SLA) is 3 to 5 business days for metro cities. Non-metro locations may require 5 to 7 business days depending on logistics service availability.",
    "doc_7": "Express Delivery SLA guarantees delivery within 24 to 48 hours for select pin codes and eligible items. Additional express shipping charges apply at checkout.",
    "doc_8": "Reverse pickup eligibility requires the pickup pincode to match the original delivery pincode. Items must be securely packed in original cardboard boxes with shipping label visible.",
    "doc_9": "Non-returnable items include innerwear, personal care, cosmetics, and opened software packages due to hygiene and safety guidelines. Exceptions are granted only for wrong item delivered.",
    "doc_10": "Order cancellation is permitted at zero penalty until the item is marked as Out for Delivery. Once out for delivery, cancellations must be refused at time of doorstep delivery.",
    "doc_11": "Refunds for cancelled orders placed via Flipkart Pay Later are settled immediately, restoring the user's available monthly credit line within 2 hours.",
    "doc_12": "High-value electronics above INR 50,000 require an Open Box Delivery verification. The delivery agent opens the package in front of the customer to verify physical condition before OTP confirmation."
}

# --- Task 10 Evaluation Ground Truth Answer Key ---
RAG_GROUND_TRUTH = [
    {"query": "How long do I have to return a pair of shoes?", "relevant_docs": ["doc_1"]},
    {"query": "When will I get my money back for a COD order?", "relevant_docs": ["doc_4"]},
    {"query": "How many days will delivery take in Bangalore?", "relevant_docs": ["doc_6", "doc_7"]},
    {"query": "Can I return a laptop after 10 days?", "relevant_docs": ["doc_2"]},
    {"query": "Can I return a product if the pickup pincode changes?", "relevant_docs": ["doc_8"]}
]

class VectorKB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.chunk_to_doc = []
        self._build_index()

    def _build_index(self):
        for doc_id, text in POLICY_DOCUMENTS.items():
            # Sentence-wise chunking
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text) if s.strip()]
            for sentence in sentences:
                self.chunks.append(sentence)
                self.chunk_to_doc.append(doc_id)
        
        embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        # L2 normalization for Cosine Similarity via Inner Product
        faiss.normalize_L2(embeddings)
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                "chunk": self.chunks[idx],
                "doc_id": self.chunk_to_doc[idx],
                "score": float(dist)
            })
        return results

if __name__ == "__main__":
    kb = VectorKB()
    print(f"Indexed {len(kb.chunks)} sentence chunks across {len(POLICY_DOCUMENTS)} documents.")