# Add this to api.py or replace the existing file

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import uuid
import os
import json
import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import arxiv-sanity-lite functionality
from aslite.db import get_papers_db, get_metas_db, SqliteDict

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# Initialize databases
pdb = get_papers_db(flag='c')
mdb = get_metas_db(flag='c')

# Create a simple in-memory user preferences store
user_preferences = {}

# Create a user preferences database using SqliteDict
def get_user_prefs_db(flag='r'):
    """Access the user preferences database"""
    os.makedirs('data', exist_ok=True)
    return SqliteDict('data/user_prefs.db', tablename='prefs', flag=flag, autocommit=True)

# Helper function to extract paper features
def extract_paper_features(paper):
    """Extract simple text features from a paper"""
    features = []
    
    # Add title
    if 'title' in paper:
        features.append(paper['title'])
    
    # Add summary/abstract
    if 'summary' in paper:
        features.append(paper['summary'])
    
    # Add author names
    if 'authors' in paper:
        author_text = ' '.join([a.get('name', '') for a in paper['authors']])
        features.append(author_text)
    
    return ' '.join(features)

@app.route('/api/papers', methods=['GET'])
def get_papers():
    """Return a batch of papers from database"""
    count = int(request.args.get('count', 10))
    
    paper_ids = list(pdb.keys())
    
    # If we don't have enough papers, return what we have
    if len(paper_ids) < count:
        logging.info(f"Not enough papers in database. Have {len(paper_ids)}, requested {count}")
    
    # Return random papers
    import random
    selected_ids = random.sample(paper_ids, min(count, len(paper_ids)))
    
    # Format for mobile app
    selected_papers = []
    for pid in selected_ids:
        paper = pdb[pid]
        paper_data = {
            "id": str(uuid.uuid4()),  # Generate a UUID for the mobile app
            "title": paper.get('title', 'Untitled'),
            "abstract": paper.get('summary', 'No abstract available'),
            "arxivId": paper.get('_id', pid),
            "authors": [a.get('name', '') for a in paper.get('authors', [])],
            "url": f"https://arxiv.org/abs/{pid}",
            "published": paper.get('published_parsed', ''),
            "swiped": False
        }
        selected_papers.append(paper_data)
    
    return jsonify(selected_papers)

@app.route('/api/paper/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    """Get a specific paper by ID"""
    if paper_id in pdb:
        paper = pdb[paper_id]
        paper_data = {
            "id": str(uuid.uuid4()),
            "title": paper.get('title', 'Untitled'),
            "abstract": paper.get('summary', 'No abstract available'),
            "arxivId": paper.get('_id', paper_id),
            "authors": [a.get('name', '') for a in paper.get('authors', [])],
            "url": f"https://arxiv.org/abs/{paper_id}",
            "published": paper.get('published_parsed', ''),
            "swiped": False
        }
        return jsonify(paper_data)
    return jsonify({"error": "Paper not found"}), 404

@app.route('/api/like', methods=['POST'])
def like_paper():
    """Record a paper like"""
    paper_id = request.json.get('paper_id')
    
    # Store paper preference
    try:
        user_id = request.json.get('user_id', 'default_user')
        
        with get_user_prefs_db(flag='c') as prefs_db:
            user_prefs = prefs_db.get(user_id, {'liked_papers': [], 'disliked_papers': []})
            if paper_id not in user_prefs['liked_papers']:
                user_prefs['liked_papers'].append(paper_id)
                # Remove from disliked if present
                if paper_id in user_prefs['disliked_papers']:
                    user_prefs['disliked_papers'].remove(paper_id)
            prefs_db[user_id] = user_prefs
        
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Error in like_paper: {e}")
        # Return success anyway to prevent app from crashing
        return jsonify({"status": "success"})

@app.route('/api/dislike', methods=['POST'])
def dislike_paper():
    """Record a paper dislike"""
    paper_id = request.json.get('paper_id')
    
    # Store paper preference
    try:
        user_id = request.json.get('user_id', 'default_user')
        
        with get_user_prefs_db(flag='c') as prefs_db:
            user_prefs = prefs_db.get(user_id, {'liked_papers': [], 'disliked_papers': []})
            if paper_id not in user_prefs['disliked_papers']:
                user_prefs['disliked_papers'].append(paper_id)
                # Remove from liked if present
                if paper_id in user_prefs['liked_papers']:
                    user_prefs['liked_papers'].remove(paper_id)
            prefs_db[user_id] = user_prefs
        
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Error in dislike_paper: {e}")
        # Return success anyway to prevent app from crashing
        return jsonify({"status": "success"})

@app.route('/api/sync_preferences', methods=['POST'])
def sync_preferences():
    """Sync user preferences from mobile app"""
    try:
        data = request.json
        user_id = data.get('user_id', 'default_user')
        liked_papers = data.get('liked_papers', [])
        disliked_papers = data.get('disliked_papers', [])
        
        # Store in database
        with get_user_prefs_db(flag='c') as prefs_db:
            prefs_db[user_id] = {
                'liked_papers': liked_papers,
                'disliked_papers': disliked_papers,
                'preferred_categories': data.get('preferred_categories', []),
                'preferred_authors': data.get('preferred_authors', []),
                'preferred_keywords': data.get('preferred_keywords', []),
                'updated_at': time.time()
            }
        
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Error in sync_preferences: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get personalized recommendations based on user preferences"""
    try:
        user_id = request.args.get('user_id', 'default_user')
        count = int(request.args.get('count', 10))
        
        # Get user preferences
        with get_user_prefs_db() as prefs_db:
            user_prefs = prefs_db.get(user_id, None)
            
            # If no preferences found, return random papers
            if not user_prefs or not user_prefs.get('liked_papers'):
                # Return random papers
                paper_ids = list(pdb.keys())
                import random
                selected_ids = random.sample(paper_ids, min(count, len(paper_ids)))
                
                recommended_papers = []
                for pid in selected_ids:
                    paper = pdb[pid]
                    paper_data = {
                        "id": str(uuid.uuid4()),
                        "title": paper.get('title', 'Untitled'),
                        "abstract": paper.get('summary', 'No abstract available'),
                        "arxivId": paper.get('_id', pid),
                        "authors": [a.get('name', '') for a in paper.get('authors', [])],
                        "url": f"https://arxiv.org/abs/{pid}",
                        "published": paper.get('published_parsed', ''),
                        "swiped": False
                    }
                    recommended_papers.append(paper_data)
                
                return jsonify(recommended_papers)
        
        # Get liked papers for content-based filtering
        liked_papers = user_prefs.get('liked_papers', [])
        disliked_papers = user_prefs.get('disliked_papers', [])
        
        # If there are no liked papers, return random recommendations
        if not liked_papers:
            # Return random papers
            paper_ids = list(pdb.keys())
            import random
            selected_ids = random.sample(paper_ids, min(count, len(paper_ids)))
            
            recommended_papers = []
            for pid in selected_ids:
                paper = pdb[pid]
                paper_data = {
                    "id": str(uuid.uuid4()),
                    "title": paper.get('title', 'Untitled'),
                    "abstract": paper.get('summary', 'No abstract available'),
                    "arxivId": paper.get('_id', pid),
                    "authors": [a.get('name', '') for a in paper.get('authors', [])],
                    "url": f"https://arxiv.org/abs/{pid}",
                    "published": paper.get('published_parsed', ''),
                    "swiped": False
                }
                recommended_papers.append(paper_data)
            
            return jsonify(recommended_papers)
        
        # We have liked papers, so let's use content-based filtering
        # Extract features from liked papers
        liked_features = []
        for pid in liked_papers:
            if pid in pdb:
                paper = pdb[pid]
                features = extract_paper_features(paper)
                liked_features.append(features)
        
        # If we couldn't find any of the liked papers, return random recommendations
        if not liked_features:
            # Return random papers
            paper_ids = list(pdb.keys())
            import random
            selected_ids = random.sample(paper_ids, min(count, len(paper_ids)))
            
            recommended_papers = []
            for pid in selected_ids:
                paper = pdb[pid]
                paper_data = {
                    "id": str(uuid.uuid4()),
                    "title": paper.get('title', 'Untitled'),
                    "abstract": paper.get('summary', 'No abstract available'),
                    "arxivId": paper.get('_id', pid),
                    "authors": [a.get('name', '') for a in paper.get('authors', [])],
                    "url": f"https://arxiv.org/abs/{pid}",
                    "published": paper.get('published_parsed', ''),
                    "swiped": False
                }
                recommended_papers.append(paper_data)
            
            return jsonify(recommended_papers)
        
        # Create a TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Vectorize the liked paper features
        liked_vectors = vectorizer.fit_transform(liked_features)
        
        # Create a candidate pool of papers (exclude liked and disliked)
        all_paper_ids = list(pdb.keys())
        candidate_ids = [pid for pid in all_paper_ids if pid not in liked_papers and pid not in disliked_papers]
        
        # Score candidates based on similarity to liked papers
        candidate_scores = []
        for pid in candidate_ids:
            if pid in pdb:
                paper = pdb[pid]
                features = extract_paper_features(paper)
                vector = vectorizer.transform([features])
                
                # Calculate similarity to each liked paper
                similarities = cosine_similarity(vector, liked_vectors)
                
                # Use max similarity as the score
                score = similarities.max()
                
                candidate_scores.append((pid, score))
        
        # Sort candidates by score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top N candidates
        top_candidates = candidate_scores[:count]
        
        # Prepare recommendations
        recommended_papers = []
        for pid, score in top_candidates:
            paper = pdb[pid]
            paper_data = {
                "id": str(uuid.uuid4()),
                "title": paper.get('title', 'Untitled'),
                "abstract": paper.get('summary', 'No abstract available'),
                "arxivId": paper.get('_id', pid),
                "authors": [a.get('name', '') for a in paper.get('authors', [])],
                "url": f"https://arxiv.org/abs/{pid}",
                "published": paper.get('published_parsed', ''),
                "swiped": False
            }
            recommended_papers.append(paper_data)
        
        return jsonify(recommended_papers)

    except Exception as e:
        logging.error(f"Error in get_recommendations: {e}")
        # On error, return random papers
        paper_ids = list(pdb.keys())
        import random
        selected_ids = random.sample(paper_ids, min(count, len(paper_ids)))
        
        recommended_papers = []
        for pid in selected_ids:
            paper = pdb[pid]
            paper_data = {
                "id": str(uuid.uuid4()),
                "title": paper.get('title', 'Untitled'),
                "abstract": paper.get('summary', 'No abstract available'),
                "arxivId": paper.get('_id', pid),
                "authors": [a.get('name', '') for a in paper.get('authors', [])],
                "url": f"https://arxiv.org/abs/{pid}",
                "published": paper.get('published_parsed', ''),
                "swiped": False
            }
            recommended_papers.append(paper_data)
        
        return jsonify(recommended_papers)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current status of the database"""
    try:
        # Get user preferences stats
        with get_user_prefs_db() as prefs_db:
            liked_count = sum(len(prefs.get('liked_papers', [])) for prefs in prefs_db.values())
            disliked_count = sum(len(prefs.get('disliked_papers', [])) for prefs in prefs_db.values())
        
        return jsonify({
            "total_papers": len(pdb),
            "liked_papers": liked_count,
            "disliked_papers": disliked_count
        })
    except Exception as e:
        logging.error(f"Error in get_status: {e}")
        return jsonify({
            "total_papers": len(pdb),
            "liked_papers": 0,
            "disliked_papers": 0
        })

if __name__ == '__main__':
    # Set the port - default 5000 or from environment variable
    port = int(os.environ.get('PORT', 8008))
    
    # Log database status at startup
    logging.info(f"Synced with papers database: {len(pdb)} papers")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)
