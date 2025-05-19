# Minimal test for HNSWPQ - M_pq = 2
import faiss
print(f"Faiss version: {faiss.__version__}")

d_test = 10
m_pq_test = 2 # 10 % 2 == 0, dsub = 5
M_hnsw_test = 32 # Standard HNSW connectivity

print(f"Attempting: faiss.IndexHNSWPQ({d_test}, {M_hnsw_test}, {m_pq_test})")
try:
    index = faiss.IndexHNSWPQ(d_test, M_hnsw_test, m_pq_test)
    print("Successfully created IndexHNSWPQ.")
except RuntimeError as e:
    print(f"Error creating IndexHNSWPQ (d={d_test}, M_pq={m_pq_test}): {e}")
except Exception as e_gen:
    print(f"Generic error creating IndexHNSWPQ (d={d_test}, M_pq={m_pq_test}): {e_gen}")

print(f"Faiss version: {faiss.__version__}")

d_test = 10
m_pq_test = 1 # 10 % 1 == 0, dsub = 10
M_hnsw_test = 32 # Standard HNSW connectivity

print(f"Attempting: faiss.IndexHNSWPQ({d_test}, {M_hnsw_test}, {m_pq_test})")
try:
    index = faiss.IndexHNSWPQ(d_test, M_hnsw_test, m_pq_test)
    print("Successfully created IndexHNSWPQ.")
except RuntimeError as e:
    print(f"Error creating IndexHNSWPQ (d={d_test}, M_pq={m_pq_test}): {e}")
except Exception as e_gen:
    print(f"Generic error creating IndexHNSWPQ (d={d_test}, M_pq={m_pq_test}): {e_gen}")
