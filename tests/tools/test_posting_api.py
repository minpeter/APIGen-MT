import pytest
import json
from tools.posting_api import PostingAPI


@pytest.fixture
def posting_api():
    initial_config = {
        "authenticated": True,
        "tweet_counter": 10,
        "tweets": {
            "0": {
                "id": 0,
                "username": "genealogy_enthusiast",
                "content": "Excited to start my genealogy journey!",
                "tags": ["#genealogy", "#familyhistory", "#beginnings"],
                "mentions": []
            },
            "1": {
                "id": 1,
                "username": "genealogy_enthusiast",
                "content": "Researching family history is so rewarding.",
                "tags": ["#genealogy", "#research", "#familyhistory"],
                "mentions": []
            },
            "2": {
                "id": 2,
                "username": "genealogy_enthusiast",
                "content": "Can't wait to uncover new stories about my ancestors.",
                "tags": ["#ancestors", "#familystories", "#discovery"],
                "mentions": []
            },
            "3": {
                "id": 3,
                "username": "genealogy_enthusiast",
                "content": "Genealogy is like a puzzle waiting to be solved.",
                "tags": ["#genealogy", "#puzzle", "#research"],
                "mentions": []
            },
            "4": {
                "id": 4,
                "username": "genealogy_enthusiast",
                "content": "Every family has a story worth telling.",
                "tags": ["#familystories", "#heritage", "#genealogy"],
                "mentions": []
            },
            "5": {
                "id": 5,
                "username": "genealogy_enthusiast",
                "content": "Exploring my roots is a journey of self-discovery.",
                "tags": ["#roots", "#selfdiscovery", "#familyhistory"],
                "mentions": []
            },
            "6": {
                "id": 6,
                "username": "genealogy_enthusiast",
                "content": "Family history is a treasure trove of stories.",
                "tags": ["#familyhistory", "#stories", "#heritage"],
                "mentions": []
            },
            "7": {
                "id": 7,
                "username": "genealogy_enthusiast",
                "content": "Connecting with my past to understand my present.",
                "tags": ["#connection", "#pastpresent", "#genealogy"],
                "mentions": []
            },
            "8": {
                "id": 8,
                "username": "genealogy_enthusiast",
                "content": "Genealogy: where history meets personal stories.",
                "tags": ["#genealogy", "#history", "#personalstories"],
                "mentions": []
            },
            "9": {
                "id": 9,
                "username": "genealogy_enthusiast",
                "content": "Uncovering the past, one ancestor at a time.",
                "tags": ["#ancestors", "#research", "#familyhistory"],
                "mentions": []
            }
        },
        "username": "genealogy_enthusiast",
        "password": "testpass"
    }
    return PostingAPI(initial_config)


@pytest.fixture
def unauthenticated_api():
    initial_config = {
        "authenticated": False,
        "tweet_counter": 10,
        "tweets": {
            "0": {
                "id": 0,
                "username": "genealogy_enthusiast",
                "content": "Excited to start my genealogy journey!",
                "tags": ["#genealogy", "#familyhistory", "#beginnings"],
                "mentions": []
            }
        },
        "username": "genealogy_enthusiast",
        "password": "testpass"
    }
    return PostingAPI(initial_config)


def test_authenticate_twitter_success(posting_api):
    result = posting_api.authenticate_twitter(
        username='genealogy_enthusiast',
        password='testpass'
    )
    assert result.get("authentication_status") is True


def test_authenticate_twitter_wrong_password(posting_api):
    result = posting_api.authenticate_twitter(
        username='genealogy_enthusiast',
        password='wrongpassword'
    )
    assert result.get("authentication_status") is False


def test_authenticate_twitter_wrong_username(posting_api):
    result = posting_api.authenticate_twitter(
        username='dr_smith',
        password='testpass'
    )
    assert result.get("authentication_status") is False


def test_comment_success(posting_api):
    result = posting_api.comment(
        tweet_id=0,
        comment_content='Another successful task completed today!'
    )
    assert "successfully" in result.get("comment_status", "").lower()


def test_comment_nonexistent_tweet(posting_api):
    result = posting_api.comment(
        tweet_id=999,
        comment_content='This should fail'
    )
    assert "not found" in result.get("comment_status", "").lower()


def test_comment_empty_content(posting_api):
    result = posting_api.comment(
        tweet_id=1,
        comment_content=''
    )
    assert isinstance(result, dict)


def test_follow_user_success(posting_api):
    result = posting_api.follow_user(username_to_follow='history_buff')
    assert result.get("follow_status") is True


def test_follow_user_already_following(posting_api):
    posting_api.follow_user(username_to_follow='history_buff')
    result = posting_api.follow_user(username_to_follow='history_buff')
    assert result.get("follow_status") is False


def test_follow_user_unauthenticated(unauthenticated_api):
    result = unauthenticated_api.follow_user(username_to_follow='history_buff')
    assert result.get("follow_status") is False


def test_get_tweet_existing(posting_api):
    result = posting_api.get_tweet(tweet_id=0)
    assert result.get("id") == 0


def test_get_tweet_nonexistent(posting_api):
    result = posting_api.get_tweet(tweet_id=999)
    assert result.get("id") == 0 and result.get("content") == ""


def test_get_tweet_invalid_id(posting_api):
    result = posting_api.get_tweet(tweet_id=-1)
    assert result.get("id") == 0 and result.get("content") == ""


def test_get_tweet_comments_existing(posting_api):
    posting_api.comment(tweet_id=0, comment_content='Great post!')
    result = posting_api.get_tweet_comments(tweet_id=0)
    comments = result.get("comments", [])
    assert isinstance(comments, list) and len(comments) > 0


def test_get_tweet_comments_no_comments(posting_api):
    result = posting_api.get_tweet_comments(tweet_id=1)
    comments = result.get("comments", [])
    assert isinstance(comments, list) and len(comments) == 0


def test_get_tweet_comments_nonexistent_tweet(posting_api):
    result = posting_api.get_tweet_comments(tweet_id=999)
    assert result.get("comments", []) == []


def test_get_user_stats_existing(posting_api):
    result = posting_api.get_user_stats(username='genealogy_enthusiast')
    assert isinstance(result, dict) and result.get("tweet_count", 0) > 0


def test_get_user_stats_nonexistent(posting_api):
    result = posting_api.get_user_stats(username='nonexistent_user')
    assert isinstance(result, dict)


def test_get_user_stats_empty_username(posting_api):
    result = posting_api.get_user_stats(username='')
    assert isinstance(result, dict)


def test_get_user_tweets_existing(posting_api):
    result = posting_api.get_user_tweets(username='genealogy_enthusiast')
    tweets = result.get("user_tweets", [])
    assert isinstance(tweets, list) and len(tweets) > 0


def test_get_user_tweets_nonexistent(posting_api):
    result = posting_api.get_user_tweets(username='nonexistent_user')
    tweets = result.get("user_tweets", [])
    assert isinstance(tweets, list) and len(tweets) == 0


def test_get_user_tweets_empty_username(posting_api):
    result = posting_api.get_user_tweets(username='')
    assert isinstance(result, dict)


def test_mention_success(posting_api):
    result = posting_api.mention(
        tweet_id=1,
        mentioned_usernames=['@technewsworld']
    )
    assert "successfully" in result.get("mention_status", "").lower()


def test_mention_nonexistent_tweet(posting_api):
    result = posting_api.mention(
        tweet_id=999,
        mentioned_usernames=['@technewsworld']
    )
    assert "not found" in result.get("mention_status", "").lower()


def test_mention_empty_usernames(posting_api):
    result = posting_api.mention(
        tweet_id=0,
        mentioned_usernames=[]
    )
    assert isinstance(result, dict)


def test_post_tweet_success(posting_api):
    result = posting_api.post_tweet(
        content='Managed to archive important data files!',
        tags=['#DataManagement', '#Efficiency']
    )
    assert result.get("id") is not None and result.get("content") != ""


def test_post_tweet_with_mentions_and_tags(posting_api):
    result = posting_api.post_tweet(
        content='Initial report content More unsorted data Unsorted data',
        mentions=['@Julia'],
        tags=['#currenttechtrend']
    )
    assert result.get("id") is not None and result.get("content") != ""


def test_post_tweet_unauthenticated(unauthenticated_api):
    result = unauthenticated_api.post_tweet(
        content='This should fail'
    )
    assert result.get("id") == 0 and result.get("content") == ""


def test_retweet_success(posting_api):
    result = posting_api.retweet(tweet_id=5)
    assert "successfully" in result.get("retweet_status", "").lower()


def test_retweet_nonexistent_tweet(posting_api):
    result = posting_api.retweet(tweet_id=999)
    assert "not found" in result.get("retweet_status", "").lower()


def test_retweet_unauthenticated(unauthenticated_api):
    result = unauthenticated_api.retweet(tweet_id=0)
    assert "not authenticated" in result.get("retweet_status", "").lower()


def test_search_tweets_with_results(posting_api):
    result = posting_api.search_tweets(keyword='genealogy')
    tweets = result.get("matching_tweets", [])
    assert isinstance(tweets, list) and len(tweets) > 0


def test_search_tweets_no_results(posting_api):
    result = posting_api.search_tweets(keyword='quantum_physics')
    tweets = result.get("matching_tweets", [])
    assert isinstance(tweets, list) and len(tweets) == 0


def test_search_tweets_empty_keyword(posting_api):
    result = posting_api.search_tweets(keyword='')
    assert isinstance(result, dict)


def test_unfollow_user_success(posting_api):
    posting_api.follow_user(username_to_follow='history_buff')
    result = posting_api.unfollow_user(username_to_unfollow='history_buff')
    assert result.get("unfollow_status") is True


def test_unfollow_user_not_following(posting_api):
    result = posting_api.unfollow_user(username_to_unfollow='never_followed')
    assert result.get("unfollow_status") is False


def test_unfollow_user_unauthenticated(unauthenticated_api):
    result = unauthenticated_api.unfollow_user(username_to_unfollow='history_buff')
    assert result.get("unfollow_status") is False
