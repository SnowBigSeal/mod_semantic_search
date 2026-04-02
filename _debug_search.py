import numpy as np, config, db
from web.app import _embed, _unpack, _cosine, _rerank

cfg = config.load()
embed_url = cfg['inference']['embedding_url'].rstrip('/')
rerank_url = cfg['inference']['reranker_url'].rstrip('/')

query = 'applied energistics'
query_vec = _embed(query, embed_url)

conn = db.connect()
rows = conn.execute("""
    SELECT p.id, p.slug, p.title, p.description, p.body, e.vector
    FROM projects p JOIN embeddings e ON e.project_id = p.id
    WHERE p.loader = ? AND p.mc_version = ?
""", ('neoforge', '1.21.1')).fetchall()
conn.close()

vectors = np.stack([_unpack(r['vector']) for r in rows])
scores = _cosine(query_vec, vectors)
sorted_idx = np.argsort(scores)[::-1]

print('=== Top 15 cosine ===')
for rank, idx in enumerate(sorted_idx[:15], 1):
    print(f'{rank}. {scores[idx]:.4f}  {rows[idx]["slug"]}')

ae2_positions = [i for i,r in enumerate(rows) if 'energistics' in r['slug'].lower() or r['slug'] == 'ae2']
print('\n=== AE2-related cosine positions ===')
for i in ae2_positions:
    cosine_rank = int(np.where(sorted_idx == i)[0][0]) + 1
    print(f'  {rows[i]["slug"]}  rank={cosine_rank}  score={scores[i]:.4f}')

# Now rerank top 50
top50_idx = sorted_idx[:50]
candidates = [rows[i] for i in top50_idx]
docs = [(r['title'] + '\n' + (r['description'] or '') + '\n' + (r['body'] or '')[:500]).strip() for r in candidates]
print('\n=== Reranking top 50 ===')
rerank_scores = _rerank(query, docs, rerank_url)
reranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
for rank, (r, s) in enumerate(reranked[:15], 1):
    print(f'{rank}. {s:.4f}  {r["slug"]}')
