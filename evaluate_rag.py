from policy_kb import VectorKB, RAG_GROUND_TRUTH

def evaluate_rag():
    kb = VectorKB()
    k = 3
    
    p_at_3_list = []
    r_at_3_list = []
    
    print("=== TASK 10: RETRIEVAL EVALUATION (PRECISION@3 & RECALL@3) ===\n")
    
    for idx, item in enumerate(RAG_GROUND_TRUTH, 1):
        query = item["query"]
        ground_truth_docs = set(item["relevant_docs"])
        
        # Search Top K chunks
        retrieved = kb.search(query, top_k=k)
        
        # Deduplicate retrieved chunks back to parent document IDs
        retrieved_docs = []
        for r in retrieved:
            doc_id = r["doc_id"]
            if doc_id not in retrieved_docs:
                retrieved_docs.append(doc_id)
                
        # Calculate intersection
        relevant_retrieved = [doc for doc in retrieved_docs if doc in ground_truth_docs]
        
        precision = len(relevant_retrieved) / len(retrieved_docs) if retrieved_docs else 0.0
        recall = len(relevant_retrieved) / len(ground_truth_docs) if ground_truth_docs else 0.0
        
        p_at_3_list.append(precision)
        r_at_3_list.append(recall)
        
        print(f"Query {idx}: '{query}'")
        print(f"  Ground Truth Docs : {list(ground_truth_docs)}")
        print(f"  Retrieved Docs    : {retrieved_docs}")
        print(f"  Relevant Found    : {relevant_retrieved}")
        print(f"  Precision@3 Calc  : {len(relevant_retrieved)} / {len(retrieved_docs)} = {precision:.4f}")
        print(f"  Recall@3 Calc     : {len(relevant_retrieved)} / {len(ground_truth_docs)} = {recall:.4f}")
        print("-" * 50)
        
    mean_p = sum(p_at_3_list) / len(p_at_3_list)
    mean_r = sum(r_at_3_list) / len(r_at_3_list)
    
    print(f"\nFINAL SUMMARY RESULTS:")
    print(f"Mean Precision@3 : {mean_p:.4f}")
    print(f"Mean Recall@3    : {mean_r:.4f}")

if __name__ == "__main__":
    evaluate_rag()