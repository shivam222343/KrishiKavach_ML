"""
Advanced YouTube Semantic Search with Sentence Transformers
Uses state-of-the-art NLP models for accurate video recommendations
"""

import re
from typing import List, Dict, Optional
import os
from datetime import datetime
import httpx

# ── Monkeypatch httpx ────────────────────────────────────────────────────────
# Fix for youtube-search-python compatibility with httpx 0.28+
# The library tries to pass 'proxies' to httpx.post, which is no longer supported.
_original_post = httpx.post
def _patched_post(*args, **kwargs):
    if 'proxies' in kwargs:
        # Newer httpx handles proxies via Client initialization, not individual requests.
        # We strip it here to avoid the TypeError.
        kwargs.pop('proxies', None)
    return _original_post(*args, **kwargs)
httpx.post = _patched_post
print("🔧 [System] Applied httpx.post monkeypatch for YouTube search compatibility")

# Try to import YouTube API (optional)
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    YOUTUBE_API_AVAILABLE = True
except ImportError:
    print("⚠️ google-api-python-client not installed. YouTube API features disabled.")
    YOUTUBE_API_AVAILABLE = False
    HttpError = Exception  # Fallback

# Lightweight semantic search model
LIGHT_MODEL_NAME = 'all-MiniLM-L6-v2'
_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        print(f"⏳ Loading Semantic Search Model ({LIGHT_MODEL_NAME})...")
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer(LIGHT_MODEL_NAME)
    return _semantic_model

# YouTube API configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')

class YouTubeSemanticSearch:
    def __init__(self):
        self.youtube = None
        if YOUTUBE_API_AVAILABLE and YOUTUBE_API_KEY:
            try:
                self.youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                print("✅ YouTube Data API initialized")
            except Exception as e:
                print(f"⚠️ YouTube API initialization failed: {e}")
    
    def parse_duration(self, duration_str: str) -> int:
        if not duration_str: return 0
        if 'PT' in duration_str: # ISO 8601
            hours = re.search(r'(\d+)H', duration_str)
            minutes = re.search(r'(\d+)M', duration_str)
            seconds = re.search(r'(\d+)S', duration_str)
            total = 0
            if hours: total += int(hours.group(1)) * 60
            if minutes: total += int(minutes.group(1))
            if seconds: total += int(seconds.group(1)) / 60
            return round(total, 1)
        # Handle MM:SS or HH:MM:SS
        parts = duration_str.split(':')
        if len(parts) == 2: return int(parts[0]) + int(parts[1])/60
        if len(parts) == 3: return int(parts[0])*60 + int(parts[1]) + int(parts[2])/60
        return 0

    def parse_views(self, view_count: str) -> int:
        try: return int(str(view_count).replace(',', ''))
        except: return 0

    def calculate_engagement_score(self, views: int, likes: int, published_days_ago: int) -> float:
        if published_days_ago == 0: published_days_ago = 1
        views_per_day = views / published_days_ago
        like_ratio = (likes / views * 1000) if views > 0 else 0
        view_score = min(views_per_day / 10000, 1.0)
        like_score = min(like_ratio / 50, 1.0)
        return (view_score * 0.7 + like_score * 0.3)

    def get_days_since_published(self, published_at: str) -> int:
        try:
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            now = datetime.now(pub_date.tzinfo)
            delta = now - pub_date
            return max(delta.days, 1)
        except: return 365

    def is_animated_content(self, title: str, description: str) -> bool:
        animated_keywords = ['animated', 'animation', 'visual', 'explained', 'graphics', 'diagram', 'whiteboard']
        text = f"{title} {description}".lower()
        return any(k in text for k in animated_keywords)

    def is_coding_content(self, title: str, description: str) -> bool:
        coding_keywords = ['code', 'coding', 'tutorial', 'implementation', 'step by step', 'guide']
        text = f"{title} {description}".lower()
        return any(k in text for k in coding_keywords)

    def is_irrelevant_health_content(self, title: str, description: str) -> bool:
        # Strict blacklist for human health terms that often collide with "yield/growth"
        health_blacklist = [
            'sperm', 'fertility', 'pregnancy', 'erectile', 'testosterone', 'sexual', 
            'motility', 'human health', 'doctor', 'patient', 'medicine', 'clinic',
            'weight loss', 'muscle', 'gym', 'workout', 'bodybuilding'
        ]
        text = f"{title} {description}".lower()
        return any(k in text for k in health_blacklist)

    def search_youtube_api(self, query: str, max_results: int = 30) -> List[Dict]:
        if not self.youtube: return []
        try:
            search_response = self.youtube.search().list(
                q=query, part='id,snippet', maxResults=max_results, type='video', safeSearch='strict'
            ).execute()
            video_ids = [item['id']['videoId'] for item in search_response.get('items', [])]
            if not video_ids: return []
            videos_response = self.youtube.videos().list(part='snippet,contentDetails,statistics', id=','.join(video_ids)).execute()
            videos = []
            for item in videos_response.get('items', []):
                snippet = item['snippet']
                stats = item.get('statistics', {})
                videos.append({
                    'id': item['id'],
                    'title': snippet['title'],
                    'description': snippet.get('description', ''),
                    'thumbnail': snippet['thumbnails']['high']['url'],
                    'channel': snippet['channelTitle'],
                    'published_at': snippet['publishedAt'],
                    'duration': item['contentDetails']['duration'],
                    'duration_minutes': self.parse_duration(item['contentDetails']['duration']),
                    'views': self.parse_views(stats.get('viewCount', '0')),
                    'likes': self.parse_views(stats.get('likeCount', '0'))
                })
            return videos
        except: return []

    def search_with_fallback(self, query: str, max_results: int = 30) -> List[Dict]:
        print(f"🔍 [YouTube Fallback] Searching for: '{query}'")
        videos = self.search_youtube_api(query, max_results)
        if videos: 
            print(f"✅ [YouTube API] Found {len(videos)} videos")
            return videos
            
        try:
            from youtubesearchpython import VideosSearch
            video_search = VideosSearch(query, limit=max_results)
            results = video_search.result()
            videos = []
            raw_results = results.get('result', [])
            print(f"📡 [youtubesearchpython] Raw results found: {len(raw_results)}")
            
            for video in raw_results:
                videos.append({
                    'id': video.get('id', ''),
                    'title': video.get('title', ''),
                    'description': video.get('descriptionSnippet', [{}])[0].get('text', '') if video.get('descriptionSnippet') else '',
                    'thumbnail': video.get('thumbnails', [{}])[0].get('url', ''),
                    'channel': video.get('channel', {}).get('name', ''),
                    'published_at': video.get('publishedTime', ''),
                    'duration': video.get('duration', '0:00'),
                    'duration_minutes': self.parse_duration(video.get('duration', '0:00')),
                    'views': self.parse_views(video.get('viewCount', {}).get('text', '0').split(' ')[0])
                })
            return videos
        except ImportError:
            print("❌ [YouTube Fallback] 'youtube-search-python' not installed. Please run: pip install youtube-search-python")
            return []
        except Exception as e:
            print(f"❌ [YouTube Fallback] Search error: {e}")
            return []

    def semantic_search(self, query: str, max_duration_minutes: int = 20, language: str = 'english') -> List[Dict]:
        model = get_semantic_model()
        from sentence_transformers import util
        # Normalize language name
        language = language.lower()
        
        # 1. Generate query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 2. Search YouTube
        # If the query is already long, don't add too many extra tokens
        search_query = query
        # Force agricultural context
        if "farming" not in search_query.lower() and "agriculture" not in search_query.lower():
            search_query = f"agriculture farming crop {search_query}"
            
        if language not in search_query.lower():
            search_query += f" {language}"
        
        print(f"🚀 [Semantic Search] Executing fallback fetch with query: '{search_query}'")
        videos = self.search_with_fallback(search_query, max_results=30)
        
        if not videos:
            print(f"⚠️ [Semantic Search] No videos returned from fallback search for: '{search_query}'")
            return []
        
        print(f"🧠 [Semantic Search] Scoring {len(videos)} videos...")
        # 3. Score and Rank
        scored_videos = []
        for video in videos:
            # 1. Block Irrelevant Content
            if self.is_irrelevant_health_content(video['title'], video['description']):
                print(f"🚫 [Filter] Blocking potential human health video: {video['title']}")
                continue

            # Generate video embedding
            video_text = f"{video['title']} {video['description'][:300]}"
            video_embedding = model.encode(video_text, convert_to_tensor=True)
            
            # Semantic Similarity
            semantic_score = util.cos_sim(query_embedding, video_embedding).item()
            
            # Recency Score
            days_ago = self.get_days_since_published(video.get('published_at', '')) if video.get('published_at') else 365
            recency_score = 1.0 if days_ago < 365 else max(0.2, 1 - (days_ago / 3650)) # Prefer last 1 year
            
            # View/Popularity Score
            view_score = min(video['views'] / 500000, 1.0) # 500k views = max
            
            # Duration Score (Prefer 10-20 min as requested)
            duration = video['duration_minutes']
            if 10 <= duration <= 20: duration_score = 1.0
            elif 5 <= duration <= 25: duration_score = 0.7
            else: duration_score = 0.4
            
            # Language Match (Simple check in title/desc)
            lang_match = 1.2 if language in video['title'].lower() or language in video['description'].lower() else 1.0

            # Final Weighted Score
            final_score = (
                semantic_score * 0.50 +
                view_score * 0.20 +
                recency_score * 0.15 +
                duration_score * 0.15
            ) * lang_match
            
            scored_videos.append({ **video, 'final_score': final_score })
            
        # Sort by final score
        scored_videos.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_videos[:6]

_youtube_search_instance = None
def get_youtube_search_instance():
    global _youtube_search_instance
    if _youtube_search_instance is None:
        _youtube_search_instance = YouTubeSemanticSearch()
    return _youtube_search_instance

def search_videos(query: str, max_duration_minutes: int = 20, language: str = 'english') -> List[Dict]:
    return get_youtube_search_instance().semantic_search(query, max_duration_minutes, language)
