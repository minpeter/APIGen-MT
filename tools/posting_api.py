"""Stateful social-posting API."""

from typing import TypedDict

from .type_utils import (
    Config,
    Record,
    get_bool,
    get_int,
    get_record_list,
    get_record_map,
    get_str,
    get_string_list,
    is_object_list,
    is_record,
)


class AuthenticationResult(TypedDict):
    """Result from social-platform authentication."""

    authentication_status: bool


class TextStatusResult(TypedDict):
    """Result carrying a descriptive operation status."""

    comment_status: str


class FollowResult(TypedDict):
    """Result from following a user."""

    follow_status: bool


class UnfollowResult(TypedDict):
    """Result from unfollowing a user."""

    unfollow_status: bool


class MentionResult(TypedDict):
    """Result from mentioning users in a post."""

    mention_status: str


class RetweetResult(TypedDict):
    """Result from reposting a post."""

    retweet_status: str


class Tweet(TypedDict):
    """Public representation of one post."""

    id: int
    username: str
    content: str
    tags: list[str]
    mentions: list[str]


class TweetComment(TypedDict):
    """Public representation of one post comment."""

    username: str
    content: str


class CommentsResult(TypedDict):
    """Comments attached to a post."""

    comments: list[Record]


class UserStatsResult(TypedDict):
    """Posting statistics for one user."""

    tweet_count: int
    following_count: int
    retweet_count: int


class UserTweetsResult(TypedDict):
    """Posts authored by one user."""

    user_tweets: list[Tweet]


class SearchTweetsResult(TypedDict):
    """Posts matching a keyword."""

    matching_tweets: list[Tweet]


def _tweet(record: Config) -> Tweet:
    return {
        "id": get_int(record, "id"),
        "username": get_str(record, "username"),
        "content": get_str(record, "content"),
        "tags": get_string_list(record, "tags"),
        "mentions": get_string_list(record, "mentions"),
    }


def _comments(config: Config) -> dict[str, list[Record]]:
    value = config.get("comments")
    if not is_record(value):
        return {}
    comments: dict[str, list[Record]] = {}
    for key, entries in value.items():
        if is_object_list(entries):
            comments[key] = [
                entry for entry in entries if is_record(entry)
            ]
    return comments


class PostingAPI:
    """Post, retrieve, and interact with social-platform messages."""

    def __init__(self, initial_config: Config) -> None:
        """Initialize the API with the given configuration."""
        nested = initial_config.get("TwitterAPI")
        config = nested if is_record(nested) else initial_config
        self.authenticated: bool = get_bool(config, "authenticated")
        self.tweet_counter: int = get_int(config, "tweet_counter")
        self.tweets: dict[str, Record] = get_record_map(config, "tweets")
        self.comments: dict[str, list[Record]] = _comments(config)
        self.retweets: list[Record] = get_record_list(config, "retweets")
        self.following_list: list[str] = get_string_list(
            config, "following_list"
        )
        self.users: dict[str, Record] = get_record_map(config, "users")
        self.username: str = get_str(config, "username")
        self.password: str = get_str(config, "password")

    def authenticate_twitter(
        self, username: str, password: str
    ) -> AuthenticationResult:
        """Authenticate a user with a username and password."""
        success = bool(username and password)
        success = success and username == self.username
        success = success and password == self.password
        if success:
            self.authenticated = True
        return {"authentication_status": success}

    def comment(self, tweet_id: int, comment_content: str) -> TextStatusResult:
        """Comment on a post as the authenticated user."""
        if not self.authenticated:
            return {"comment_status": "User not authenticated"}
        tweet_key = str(tweet_id)
        if tweet_key not in self.tweets:
            return {"comment_status": "Tweet not found"}
        if not comment_content:
            return {"comment_status": "Comment content cannot be empty"}
        comment: Record = {
            "username": self.username,
            "content": comment_content,
        }
        self.comments.setdefault(tweet_key, []).append(comment)
        return {"comment_status": "Comment posted successfully"}

    def follow_user(self, username_to_follow: str) -> FollowResult:
        """Follow a user as the authenticated user."""
        if (
            not self.authenticated
            or not username_to_follow
            or username_to_follow in self.following_list
        ):
            return {"follow_status": False}
        self.following_list.append(username_to_follow)
        return {"follow_status": True}

    def get_tweet(self, tweet_id: int) -> Tweet:
        """Retrieve a specific post."""
        record = self.tweets.get(str(tweet_id))
        return _tweet(record) if record is not None else _tweet({})

    def get_tweet_comments(self, tweet_id: int) -> CommentsResult:
        """Retrieve all comments for a specific post."""
        return {"comments": self.comments.get(str(tweet_id), [])}

    def get_user_stats(self, username: str) -> UserStatsResult:
        """Get posting statistics for a specific user."""
        tweet_count = sum(
            get_str(tweet, "username") == username
            for tweet in self.tweets.values()
        )
        retweet_count = sum(
            get_str(retweet, "username") == username
            for retweet in self.retweets
        )
        following_count = len(self.following_list) if username == self.username else 0
        user = self.users.get(username)
        if user is not None:
            following_count = get_int(user, "following_count", following_count)
            tweet_count = get_int(user, "tweet_count", tweet_count)
            retweet_count = get_int(user, "retweet_count", retweet_count)
        return {
            "tweet_count": tweet_count,
            "following_count": following_count,
            "retweet_count": retweet_count,
        }

    def get_user_tweets(self, username: str) -> UserTweetsResult:
        """Retrieve all posts from a specific user."""
        return {
            "user_tweets": [
                _tweet(tweet)
                for tweet in self.tweets.values()
                if get_str(tweet, "username") == username
            ]
        }

    def mention(
        self, tweet_id: int, mentioned_usernames: list[str]
    ) -> MentionResult:
        """Mention specified users in a post."""
        if not self.authenticated:
            return {"mention_status": "User not authenticated"}
        tweet = self.tweets.get(str(tweet_id))
        if tweet is None:
            return {"mention_status": "Tweet not found"}
        if not mentioned_usernames:
            return {"mention_status": "No usernames provided to mention"}
        current_mentions = get_string_list(tweet, "mentions")
        for username in mentioned_usernames:
            mention = username if username.startswith("@") else f"@{username}"
            if mention not in current_mentions:
                current_mentions.append(mention)
        tweet["mentions"] = current_mentions
        return {"mention_status": "Users mentioned successfully"}

    def post_tweet(
        self,
        content: str,
        tags: list[str] | None = None,
        mentions: list[str] | None = None,
    ) -> Tweet:
        """Post a message as the authenticated user."""
        if not self.authenticated:
            return _tweet({})
        processed_tags = [tag if tag.startswith("#") else f"#{tag}" for tag in tags or []]
        processed_mentions = [
            mention if mention.startswith("@") else f"@{mention}"
            for mention in mentions or []
        ]
        tweet_id = self.tweet_counter
        self.tweet_counter += 1
        new_tweet: Tweet = {
            "id": tweet_id,
            "username": self.username,
            "content": content,
            "tags": processed_tags,
            "mentions": processed_mentions,
        }
        self.tweets[str(tweet_id)] = dict(new_tweet)
        return new_tweet

    def retweet(self, tweet_id: int) -> RetweetResult:
        """Repost a post as the authenticated user."""
        if not self.authenticated:
            return {"retweet_status": "User not authenticated"}
        if str(tweet_id) not in self.tweets:
            return {"retweet_status": "Tweet not found"}
        self.retweets.append({"username": self.username, "tweet_id": tweet_id})
        return {"retweet_status": "Retweeted successfully"}

    def search_tweets(self, keyword: str) -> SearchTweetsResult:
        """Search for posts containing a keyword."""
        keyword = keyword.lower()
        return {
            "matching_tweets": [
                _tweet(tweet)
                for tweet in self.tweets.values()
                if keyword and keyword in get_str(tweet, "content").lower()
            ]
        }

    def unfollow_user(self, username_to_unfollow: str) -> UnfollowResult:
        """Unfollow a user as the authenticated user."""
        if (
            not self.authenticated
            or not username_to_unfollow
            or username_to_unfollow not in self.following_list
        ):
            return {"unfollow_status": False}
        self.following_list.remove(username_to_unfollow)
        return {"unfollow_status": True}
