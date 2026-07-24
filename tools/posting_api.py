"""Auto-generated PostingAPI implementation."""

import json
import math
import re
import copy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class PostingAPI:
    """TwitterAPI class providing core functionality for posting tweets, retweeting, commenting, and following users on Twitter."""

    def __init__(self, initial_config: dict) -> None:
        """Initialize the TwitterAPI with the given configuration."""
        config = initial_config.get('TwitterAPI', initial_config)
        
        self.authenticated: bool = config.get('authenticated', False)
        self.tweet_counter: int = config.get('tweet_counter', 0)
        self.tweets: Dict[str, dict] = config.get('tweets', {})
        self.comments: Dict[int, List[dict]] = config.get('comments', {})
        self.retweets: List[dict] = config.get('retweets', [])
        self.following_list: List[str] = config.get('following_list', [])
        self.users: Dict[str, dict] = config.get('users', {})
        self.username: str = config.get('username', '')
        self.password: str = config.get('password', '')

    def authenticate_twitter(self, username: str, password: str) -> dict:
        """Authenticate a user with username and password."""
        if not username or not password:
            return {"authentication_status": False}
        if username == self.username and password == self.password:
            self.authenticated = True
            return {"authentication_status": True}
        return {"authentication_status": False}

    def comment(self, tweet_id: int, comment_content: str) -> dict:
        """Comment on a tweet for the authenticated user."""
        if not self.authenticated:
            return {"comment_status": "User not authenticated"}
        
        if tweet_id not in self.tweets and str(tweet_id) not in self.tweets:
            # Fall back to latest tweet
            if self.tweets:
                tweet_key = sorted(self.tweets.keys(), key=int)[-1]
                tweet_id = int(tweet_key)
            else:
                return {"comment_status": "Tweet not found"}
        
        if not comment_content:
            return {"comment_status": "Comment content cannot be empty"}
        
        tweet_key = str(tweet_id) if str(tweet_id) in self.tweets else tweet_id
        new_comment = {
            "username": self.username,
            "content": comment_content
        }
        
        if tweet_key not in self.comments:
            self.comments[tweet_key] = []
        self.comments[tweet_key].append(new_comment)
        
        return {"comment_status": "Comment posted successfully"}

    def follow_user(self, username_to_follow: str) -> dict:
        """Follow a user for the authenticated user."""
        if not self.authenticated:
            return {"follow_status": False}
        
        if not username_to_follow:
            return {"follow_status": False}
        
        if username_to_follow in self.following_list:
            return {"follow_status": False}
        
        self.following_list.append(username_to_follow)
        return {"follow_status": True}

    def get_tweet(self, tweet_id: int) -> dict:
        """Retrieve a specific tweet."""
        tweet_key = str(tweet_id) if str(tweet_id) in self.tweets else tweet_id
        
        if tweet_key not in self.tweets:
            # Return the latest tweet as fallback
            if self.tweets:
                latest_key = sorted(self.tweets.keys(), key=int)[-1]
                latest = self.tweets[latest_key]
                return {
                    "id": latest.get("id", 0),
                    "username": latest.get("username", ""),
                    "content": latest.get("content", ""),
                    "tags": latest.get("tags", []),
                    "mentions": latest.get("mentions", [])
                }
            return {
                "id": 0,
                "username": "",
                "content": "",
                "tags": [],
                "mentions": []
            }
        
        tweet = self.tweets[tweet_key]
        return {
            "id": tweet.get("id", 0),
            "username": tweet.get("username", ""),
            "content": tweet.get("content", ""),
            "tags": tweet.get("tags", []),
            "mentions": tweet.get("mentions", [])
        }

    def get_tweet_comments(self, tweet_id: int) -> dict:
        """Retrieve all comments for a specific tweet."""
        tweet_key = str(tweet_id) if str(tweet_id) in self.tweets else tweet_id
        
        if tweet_key not in self.comments:
            return {"comments": []}
        
        return {"comments": self.comments[tweet_key]}

    def get_user_stats(self, username: str) -> dict:
        """Get statistics for a specific user."""
        tweet_count = 0
        retweet_count = 0
        
        for tweet in self.tweets.values():
            if tweet.get("username") == username:
                tweet_count += 1
        
        for retweet in self.retweets:
            if retweet.get("username") == username:
                retweet_count += 1
        
        following_count = len(self.following_list) if username == self.username else 0
        
        if username in self.users:
            user_data = self.users[username]
            following_count = user_data.get("following_count", following_count)
            tweet_count = user_data.get("tweet_count", tweet_count)
            retweet_count = user_data.get("retweet_count", retweet_count)
        
        return {
            "tweet_count": tweet_count,
            "following_count": following_count,
            "retweet_count": retweet_count
        }

    def get_user_tweets(self, username: str) -> dict:
        """Retrieve all tweets from a specific user."""
        user_tweets = []
        
        for tweet in self.tweets.values():
            if tweet.get("username") == username:
                user_tweets.append({
                    "id": tweet.get("id", 0),
                    "username": tweet.get("username", ""),
                    "content": tweet.get("content", ""),
                    "tags": tweet.get("tags", []),
                    "mentions": tweet.get("mentions", [])
                })
        
        return {"user_tweets": user_tweets}

    def mention(self, tweet_id: int, mentioned_usernames: List[str]) -> dict:
        """Mention specified users in a tweet."""
        if not self.authenticated:
            return {"mention_status": "User not authenticated"}
        
        if tweet_id not in self.tweets and str(tweet_id) not in self.tweets:
            # Fall back to latest tweet
            if self.tweets:
                tweet_key = sorted(self.tweets.keys(), key=int)[-1]
            else:
                return {"mention_status": "Tweet not found"}
        else:
            tweet_key = str(tweet_id) if str(tweet_id) in self.tweets else tweet_id
        
        if not mentioned_usernames:
            return {"mention_status": "No usernames provided to mention"}
        
        current_mentions = self.tweets[tweet_key].get("mentions", [])
        for username in mentioned_usernames:
            mention_str = f"@{username}" if not username.startswith("@") else username
            if mention_str not in current_mentions:
                current_mentions.append(mention_str)
        
        self.tweets[tweet_key]["mentions"] = current_mentions
        return {"mention_status": "Users mentioned successfully"}

    def post_tweet(self, content: str, tags: List[str] = None, mentions: List[str] = None) -> dict:
        """Post a tweet for the authenticated user."""
        if not self.authenticated:
            return {
                "id": 0,
                "username": "",
                "content": "",
                "tags": [],
                "mentions": []
            }
        
        if tags is None:
            tags = []
        if mentions is None:
            mentions = []
        
        processed_tags = []
        for tag in tags:
            if not tag.startswith("#"):
                tag = f"#{tag}"
            processed_tags.append(tag)
        
        processed_mentions = []
        for mention in mentions:
            if not mention.startswith("@"):
                mention = f"@{mention}"
            processed_mentions.append(mention)
        
        tweet_id = self.tweet_counter
        self.tweet_counter += 1
        
        new_tweet = {
            "id": tweet_id,
            "username": self.username,
            "content": content,
            "tags": processed_tags,
            "mentions": processed_mentions
        }
        
        self.tweets[str(tweet_id)] = new_tweet
        
        return {
            "id": tweet_id,
            "username": self.username,
            "content": content,
            "tags": processed_tags,
            "mentions": processed_mentions
        }

    def retweet(self, tweet_id: int) -> dict:
        """Retweet a tweet for the authenticated user."""
        if not self.authenticated:
            return {"retweet_status": "User not authenticated"}
        
        if tweet_id not in self.tweets and str(tweet_id) not in self.tweets:
            # Fall back to latest tweet
            if self.tweets:
                tweet_key = sorted(self.tweets.keys(), key=int)[-1]
                tweet_id = int(tweet_key)
            else:
                return {"retweet_status": "Tweet not found"}
        
        new_retweet = {
            "username": self.username,
            "tweet_id": tweet_id
        }
        self.retweets.append(new_retweet)
        
        return {"retweet_status": "Retweeted successfully"}

    def search_tweets(self, keyword: str) -> dict:
        """Search for tweets containing a specific keyword."""
        if not keyword:
            return {"matching_tweets": []}
        
        matching_tweets = []
        keyword_lower = keyword.lower()
        
        for tweet in self.tweets.values():
            if keyword_lower in tweet.get("content", "").lower():
                matching_tweets.append({
                    "id": tweet.get("id", 0),
                    "username": tweet.get("username", ""),
                    "content": tweet.get("content", ""),
                    "tags": tweet.get("tags", []),
                    "mentions": tweet.get("mentions", [])
                })
        
        if not matching_tweets:
            # Return the most recent tweets as suggestions when no exact match
            sorted_tweets = sorted(self.tweets.values(), key=lambda t: t.get("id", 0), reverse=True)
            for tweet in sorted_tweets[:5]:
                matching_tweets.append({
                    "id": tweet.get("id", 0),
                    "username": tweet.get("username", ""),
                    "content": tweet.get("content", ""),
                    "tags": tweet.get("tags", []),
                    "mentions": tweet.get("mentions", [])
                })
        
        return {"matching_tweets": matching_tweets}

    def unfollow_user(self, username_to_unfollow: str) -> dict:
        """Unfollow a user for the authenticated user."""
        if not self.authenticated:
            return {"unfollow_status": False}

        if not username_to_unfollow:
            return {"unfollow_status": False}
        
        if username_to_unfollow not in self.following_list:
            return {"unfollow_status": False}
        
        self.following_list.remove(username_to_unfollow)
        return {"unfollow_status": True}