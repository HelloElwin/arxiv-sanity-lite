from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import arxiv
import os
import json
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Import arxiv-sanity-lite functionality
from aslite.db import get_papers_db, get_metas_db
from aslite.arxiv import get_response, parse_response

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Create our own TF-IDF vectorizer (based on compute.py)
def get_tfidf_vectorizer(num_features=20000, min_df=5, max_df=0.1):
    return TfidfVectorizer(input='content',
                        encoding='utf-8', decode_error='replace', strip_accents='unicode',
                        lowercase=True, analyzer='word', stop_words='english',
                        token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b',
                        ngram_range=(1, 2), max_features=num_features,
                        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                        max_df=max_df, min_df=min_df)

# Load existing database or create empty one
db_path = 'data/tfidf.p'
os.makedirs('data', exist_ok=True)

if os.path.isfile(db_path):
    with open(db_path, 'rb') as f:
        data = pickle.load(f)
        vectorizer = data.get('vectorizer')
        if vectorizer is None:
            vectorizer = get_tfidf_vectorizer()
        X = data.get('x')
        papers = data.get('papers', {})
        # Patch existing papers that may be missing "arxiv_id"
        changed = False
        for pid, pdata in papers.items():
            if isinstance(pdata, dict) and 'arxiv_id' not in pdata:
                pdata['arxiv_id'] = pdata.get('_id', '')
                changed = True
        if changed:
            with open(db_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': vectorizer,
                    'x': X,
                    'pids': list(papers.keys()),
                    'papers': papers
                }, f)
        print(f"Loaded {len(papers)} papers")
else:
    # Initialize with empty data
    papers = {}
    vectorizer = get_tfidf_vectorizer()
    X = None


# Initialize databases
pdb = get_papers_db(flag='c')
mdb = get_metas_db(flag='c')

# Keep track of user likes/dislikes for recommendation
liked_papers = []
disliked_papers = []

def prune_database(max_papers=1000):
    """Keep only the most recent max_papers papers in the database"""
    global papers, X, pdb, mdb
    
    # Sort papers by date
    sorted_papers = sorted(
        papers.items(), 
        key=lambda x: x[1].get('_time', 0), 
        reverse=True
    )
    
    # Keep only the most recent max_papers
    if len(sorted_papers) > max_papers:
        # Get IDs of papers to keep
        keep_ids = [p[0] for p in sorted_papers[:max_papers]]
        
        # Create new papers dictionary with only recent papers
        pruned_papers = {pid: papers[pid] for pid in keep_ids}
        
        logging.info(f"Pruning database from {len(papers)} to {len(pruned_papers)} papers")
        
        # Replace the current papers dictionary
        papers = pruned_papers
        
        # Update the papers database
        for pid in list(pdb.keys()):
            if pid not in keep_ids:
                del pdb[pid]
                if pid in mdb:
                    del mdb[pid]
        
        # Recompute features for the pruned set
        update_vectors()
        
        logging.info(f"Database pruned to {len(papers)} papers")
        return len(papers)
    else:
        logging.info(f"No pruning needed, database has {len(papers)} papers (max: {max_papers})")
        return len(papers)

def update_vectors():
    """Update TF-IDF vectors for all papers"""
    global X, vectorizer, papers
    
    # Extract all abstracts
    abstracts = []
    paper_ids = []
    for pid, paper in papers.items():
        # Use summary field, as that's what arxiv papers use
        abstract = paper.get('summary', paper.get('abstract', ''))
        # Include title and authors in the features (similar to compute.py)
        title = paper.get('title', '')
        authors = paper.get('authors', [])
        if isinstance(authors, list):
            author_str = ' '.join([a.get('name', '') if isinstance(a, dict) else a for a in authors])
        else:
            author_str = ''
        
        feature_text = ' '.join([title, abstract, author_str])
        abstracts.append(feature_text)
        paper_ids.append(pid)
    
    if not abstracts:
        logging.warning("No abstracts to compute vectors for")
        return
    
    # Calculate TF-IDF
    X = vectorizer.fit_transform(abstracts)
    
    # Save to disk
    with open(db_path, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'x': X,
            'pids': paper_ids,
            'papers': papers
        }, f)
    
    logging.info(f"Updated vectors for {len(abstracts)} papers")

@app.route('/api/papers', methods=['GET'])
def get_papers():
    """Return a batch of random papers from database"""
    count = int(request.args.get('count', 10))
    paper_ids = list(papers.keys())
    
    # If we don't have enough papers, fetch more
    if len(paper_ids) < count:
        logging.info(f"Not enough papers in database. Have {len(paper_ids)}, requested {count}")
    
    # Return random papers
    import random
    selected_ids = random.sample(paper_ids, min(count, len(paper_ids)))
    selected_papers = [papers[pid] for pid in selected_ids]
    
    return jsonify(selected_papers)

@app.route('/api/paper/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    """Get a specific paper by ID"""
    if paper_id in papers:
        return jsonify(papers[paper_id])
    return jsonify({"error": "Paper not found"}), 404

@app.route('/api/like', methods=['POST'])
def like_paper():
    """Record a paper like and update recommendations"""
    paper_id = request.json.get('paper_id')
    if not paper_id or paper_id not in papers:
        return jsonify({"error": "Invalid paper ID"}), 400
    
    liked_papers.append(paper_id)
    
    # Update recommendations based on likes
    recommendations = get_recommendations()
    return jsonify({"status": "success", "recommendations": recommendations})

@app.route('/api/dislike', methods=['POST'])
def dislike_paper():
    """Record a paper dislike"""
    paper_id = request.json.get('paper_id')
    if not paper_id or paper_id not in papers:
        return jsonify({"error": "Invalid paper ID"}), 400
    
    disliked_papers.append(paper_id)
    return jsonify({"status": "success"})

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get paper recommendations based on liked papers"""
    # In a real implementation, use the TF-IDF vectors and SVM approach
    # from arxiv-sanity-lite to generate recommendations
    
    # For now, implement a simple recommendation based on TF-IDF similarity
    if not liked_papers or X is None:
        # If no likes yet, return random papers
        all_ids = list(papers.keys())
        import random
        count = min(10, len(all_ids))
        if count > 0:
            rec_ids = random.sample(all_ids, count)
            recommendations = [papers[pid] for pid in rec_ids]
        else:
            recommendations = []
    else:
        # Simple recommendation: find papers similar to the liked ones
        # based on TF-IDF vectors
        paper_ids = list(papers.keys())
        liked_indices = []
        for pid in liked_papers:
            if pid in papers:
                try:
                    idx = paper_ids.index(pid)
                    liked_indices.append(idx)
                except ValueError:
                    pass  # Paper ID not found in the current vectors
                    
        if not liked_indices:
            return jsonify([])
        
        # Compute average vector of liked papers
        from scipy.sparse import vstack
        liked_vecs = vstack([X[i] for i in liked_indices])
        avg_vec = liked_vecs.mean(axis=0)
        
        # Compute similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        scores = cosine_similarity(avg_vec, X)[0]
        
        # Sort by similarity and exclude already liked/disliked papers
        scored_papers = [(pid, score) for pid, score in zip(paper_ids, scores) 
                         if pid not in liked_papers and pid not in disliked_papers]
        scored_papers.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 10
        recommendations = [papers[pid] for pid, _ in scored_papers[:10]]
    
    return jsonify(recommendations)

@app.route('/api/fetch', methods=['POST'])
def fetch_papers():
    """Fetch new papers from arxiv based on search query"""
    query = request.json.get('query', 'cs.AI OR cs.LG')
    max_results = request.json.get('max_results', 100)
    max_papers = request.json.get('max_papers', 1000)  # Max papers to keep after pruning
    
    # Fetch from arxiv API
    logging.info(f"Fetching papers with query: {query}, max_results: {max_results}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    for result in client.results(search):
        paper = {
            '_id': result.entry_id.split('/')[-1],
            'title': result.title,
            'summary': result.summary,  # Using summary field to match arxiv-sanity-lite
            'authors': [{'name': author.name} for author in result.authors],  # Match format
            'categories': result.categories,
            'published': result.published.isoformat(),
            'arxivLink': result.entry_id,
            '_time': time.time()  # Add current timestamp
        }
        # Add the required "arxiv_id" key using _id as its value
        paper['arxiv_id'] = paper['_id']
        
        # Add to our database
        papers[paper['_id']] = paper
        pdb[paper['_id']] = paper
        mdb[paper['_id']] = {'_time': paper['_time']}
        results.append(paper)
    
    logging.info(f"Fetched {len(results)} papers")
    
    # Update vectors
    update_vectors()
    
    # Prune database if needed
    prune_database(max_papers)
    
    return jsonify({"status": "success", "fetched": len(results), "total_papers": len(papers)})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of the database"""
    return jsonify({
        "total_papers": len(papers),
        "liked_papers": len(liked_papers),
        "disliked_papers": len(disliked_papers)
    })

if __name__ == '__main__':
    # Set the port - default 5000 or from environment variable
    port = int(os.environ.get('PORT', 8008))
    
    # Ensure the database is in sync with pdb at startup
    if pdb:
        for pid, paper in pdb.items():
            papers[pid] = paper
        logging.info(f"Synced with papers database: {len(papers)} papers")
    
    # Update vectors if needed
    if len(papers) > 0 and (X is None or (hasattr(X, 'shape') and X.shape[0] != len(papers))):
        logging.info("Updating vectors at startup")
        update_vectors()
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)

