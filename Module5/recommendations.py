# Todd Bartoszkiewicz
# CSC506: Introduction to Data Structures and Algorithms
# Module 5: Critical Thinking
#
# Hash Table class to show some of what we learned this week.
# We could also use a Dictionary that would make this code simpler and would allow us to remove the HashTable class
# and replace it with the Python dictionary data type.
import time


# Using the Hashtable structure from 5.1 gives us:
class HashTable:

    def __init__(self):
        self.size = 256
        # Chaining to handle collisions
        self.slots = [[] for _ in range(self.size)]
        self.count = 0

    def _hash(self, key):
        multiplier = 1
        hash_value = 0
        for char in key:
            hash_value += multiplier * ord(char)
            multiplier += 1
        return hash_value % self.size

    def insert(self, key, value):
        index = self._hash(key)
        # Check if key exists, update if found
        for i, (k, _) in enumerate(self.slots[index]):
            if k == key:
                self.slots[index][i] = (key, value)
                return
        # If not found, add new key-value pair
        self.slots[index].append((key, value))
        self.count += 1

    def search(self, key):
        index = self._hash(key)
        for k, v in self.slots[index]:
            if k == key:
                return v
        return None

    def delete(self, key):
        index = self._hash(key)
        for i, (k, v) in enumerate(self.slots[index]):
            if k == key:
                del self.slots[index][i]
                self.count -= 1
                return True
        return False


class RecommendationSystem:
    def __init__(self):
        self.user_profiles = HashTable()
        self.content_data = HashTable()
        self.user_interactions = HashTable()
        print("Recommendation system initialized with hash tables.")

    def add_user(self, user_id, profile_data):
        print(f"Adding/Updating user: {user_id}")
        self.user_profiles.insert(user_id, profile_data)
        if self.user_interactions.search(user_id) is None:
            self.user_interactions.insert(user_id, set())

    def get_user_profiles(self, user_id):
        return self.user_profiles.search(user_id)

    def add_content(self, cid, data):
        print(f"Adding/Updating content: {cid}")
        self.content_data.insert(cid, data)

    def get_content_data(self, cid):
        return self.content_data.search(cid)

    def record_interaction(self, user_id, cid, interaction_type):
        print(f"Recording interaction: user {user_id} with content {cid} ({interaction_type})")
        if not self.user_interactions.search(user_id) is None:
            self.user_interactions.insert(user_id, cid)
        else:
            print(f"Warning: Interaction recorded for unknown user {user_id}")
            self.user_interactions.insert(user_id, cid)

    def has_interacted(self, user_id, cid):
        return not self.user_interactions.search(user_id) is None and cid in self.user_interactions.search(user_id)

    def generate_recommendations(self, user_id, num_recommendations):
        print(f"\nGenerating recommendations for user: {user_id}")
        user_profile = self.get_user_profiles(user_id)

        if not user_profile:
            print(f"User {user_id} not found.")
            return []

        user_interests = set(user_profile['interests'])
        if not user_interests:
            print(f"User {user_id} has no specified interests.")
            return []

        # candidate_content_ids = list(self.content_data.keys())
        candidate_content_ids = []
        for i in range(self.content_data.size):
            for k, v in self.content_data.slots[i]:
                print(k)
                candidate_content_ids.append(k)
        recommendation_scores = []

        start_time = time.time()
        for cid in candidate_content_ids:
            data = self.get_content_data(cid)
            if not data:
                continue

            if self.has_interacted(user_id, cid):
                continue

            content_tags = set(data.get('tags', []))
            temp_score = len(user_interests.intersection(content_tags))

            engagement_score = data.get('engagement_score', 0)
            combined_score = temp_score + engagement_score * 0.1

            if combined_score > 0:
                recommendation_scores.append((cid, combined_score))

        end_time = time.time()
        print(f"Scored candidates took: {end_time - start_time:.4f} seconds")

        recommendation_scores.sort(key=lambda item: item[1], reverse=True)
        top_recommendations = recommendation_scores[:num_recommendations]

        print(f"Generated {len(top_recommendations)} potential recommendations.")
        return top_recommendations


if __name__ == "__main__":
    # recommendations = HashTable()
    system = RecommendationSystem()

    # Add user recommendations
    # recommendations.insert("user123", ["post1", "post2", "post5"])
    # recommendations.insert("user456", ["post3", "post7"])
    # recommendations.insert("user789", ["post2", "post4", "post6"])
    system.add_user("user123", {"interests": ["technology", "programming", "science"], "location": "USA"})
    system.add_user("user456", {"interests": ["travel", "food", "photography"], "location": "Germany"})
    system.add_user("user789", {"interests": ["science", "history", "books"], "location": "UK"})

    system.add_content("contentA", {"tags": ["technology", "gadgets"], "title": "New Gadget Review",
                                    "engagement_score": 0.8})
    system.add_content("contentB", {"tags": ["travel", "europe", "photography"], "title": "Best Spots in Europe",
                                    "engagement_score": 0.95})
    system.add_content("contentC", {"tags": ["programming", "python", "tutorial"], "title": "Python Async Tutorial",
                                    "engagement_score": 0.7})
    system.add_content("contentD", {"tags": ["science", "astronomy"], "title": "Latest Space Discoveries",
                                    "engagement_score": 0.9})
    system.add_content("contentE", {"tags": ["food", "recipes"], "title": "Italian Pasta Recipes",
                                    "engagement_score": 0.85})
    system.add_content("contentF", {"tags": ["history", "ancient_rome"], "title": "Fall of the Roman Empire",
                                    "engagement_score": 0.75})
    system.add_content("contentG", {"tags": ["technology", "ai", "future"], "title": "The Future of AI",
                                    "engagement_score": 0.92})

    system.record_interaction("user123", "contentA", "view")
    system.record_interaction("user123", "contentC", "like")
    system.record_interaction("user456", "contentB", "view")
    system.record_interaction("user789", "contentF", "view")
    system.record_interaction("user789", "contentD", "like")

    recommendations_user123 = system.generate_recommendations("user123", 5)
    print(f"\nRecommendations for user123:")
    if recommendations_user123:
        for content_id, score in recommendations_user123:
            content_data = system.get_content_data(content_id)
            print(f"- {content_id}: {content_data['title']} (Score: {score:.2f})")
    else:
        print("No recommendations generated.")

    recommendations_user456 = system.generate_recommendations("user456", 5)
    print(f"\nRecommendations for user456:")
    if recommendations_user456:
        for content_id, score in recommendations_user456:
            content_data = system.get_content_data(content_id)
            print(f"- {content_id}: {content_data['title']} (Score: {score:.2f})")
    else:
        print("No recommendations generated.")

    recommendations_user789 = system.generate_recommendations("user789", 5)
    print(f"\nRecommendations for user789:")
    if recommendations_user789:
        for content_id, score in recommendations_user789:
            content_data = system.get_content_data(content_id)
            print(f"- {content_id}: {content_data['title']} (Score: {score:.2f})")
    else:
        print("No recommendations generated.")

    print("\n--- Adding new relevant content ---")
    system.add_content("contentH", {"tags": ["technology", "programming", "security"], "title": "Cybersecurity Trends",
                                    "engagement_score": 0.88})

    recommendations_user123_after_update = system.generate_recommendations("user123", 5)
    print(f"\nRecommendations for user123 (after adding contentH):")
    if recommendations_user123_after_update:
        for content_id, score in recommendations_user123_after_update:
            content_data = system.get_content_data(content_id)
            print(f"- {content_id}: {content_data['title']} (Score: {score:.2f})")
    else:
        print("No recommendations generated.")
